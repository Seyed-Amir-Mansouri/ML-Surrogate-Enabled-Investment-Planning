"""Build the coupled electricity + hydrogen dispatch LP with linopy."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from .config import RunConfig
from . import data_loader as dl
from . import exports_loader
from . import network_loader as nl
from .data_loader import ZoneData
from .network_loader import NetworkData, Line

HOUR = "hour"
GEN = "gen"
ZONE = "zone"
STO = "sto"


def _num(arr) -> np.ndarray:
    """Coerce to float array with NaN/inf replaced by 0 (blank profile cells)."""
    return np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class BuildResult:
    model: linopy.Model
    cfg: RunConfig
    zones: list[str]
    hours: pd.Index
    gens: pd.DataFrame
    commit: pd.DataFrame
    storage: pd.DataFrame
    gen_upper: xr.DataArray
    demand_e: xr.DataArray
    demand_h: xr.DataArray
    external_e: xr.DataArray
    external_h2: xr.DataArray
    elines: list[Line]
    hlines: list[Line]
    net: NetworkData
    price_e: xr.DataArray | None = None
    price_h: xr.DataArray | None = None
    uc_gens: list[str] | None = None
    startup_cost_eur: float = 0.0


def _marginal_cost(zd: ZoneData, tech: str, h2_fuel: bool, co2_price: float,
                   cfg: RunConfig) -> float:
    """Short-run marginal cost (EUR/MWh_elec) for a dispatchable/profile tech."""
    vom = zd.char_val(tech, "Price (EUR/MWh)", 0.0)
    fuel = zd.char_val(tech, "Fuel (EUR/MWh)", 0.0)
    co2f = zd.char_val(tech, "CO2 Factor (ton/MWh)", 0.0)
    if h2_fuel:
        return vom
    eff = zd.char_val(tech, "Efficiency (%)", 0.0) / 100.0
    e = eff if eff > 0 else cfg.default_efficiency
    fuel_term = fuel / e if cfg.fuel_per_thermal else fuel
    co2_term = (co2f / e if cfg.co2_per_thermal else co2f) * co2_price
    return vom + fuel_term + co2_term


def _build_generators(zdata: dict[str, ZoneData], net: NetworkData, cfg: RunConfig):
    """Return (gens DataFrame, per-gen hourly upper-bound array as dict)."""
    month = cfg.month_index()
    rows: list[dict] = []
    upper: dict[str, np.ndarray] = {}
    H = len(zdata[cfg.zones[0]].profiles)

    for z in cfg.zones:
        zd = zdata[z]
        for tech, cap in zd.capacities.items():
            cap = float(cap or 0.0)
            category, h2_fuel = dl.classify(tech)
            gid = f"{z}|{tech}"

            if category == dl.CAT_COMMIT:
                if cap <= 0:
                    continue
                units = int(round(zd.char_val(tech, "Number of Units", 0.0)))
                units = max(units, 1)
                max_units = units
                pmax_unit = cap / units
                msp = zd.char_val(tech, "Minimum Stable Power (%)", 0.0) / 100.0
                pmin_unit = pmax_unit * msp
                ramp_pu = zd.char_val(tech, "Ramp-Up Rate (MW/h)", 0.0)
                ramp_dn = zd.char_val(tech, "Ramp-Down Rate (MW/h)", 0.0)
                mustrun_pct = zd.must_run_pct(tech, month)
                mustrun_pct = float(min(max(mustrun_pct, 0.0), 100.0))
                pmin_floor = (mustrun_pct / 100.0) * cap if mustrun_pct > 0 else 0.0
                eff = zd.char_val(tech, "Efficiency (%)", 0.0) / 100.0
                eff = eff if eff > 1e-3 else cfg.default_efficiency
                min_up_h = zd.char_val(tech, "Minimum Up Time (h)", 0.0)
                min_down_h = zd.char_val(tech, "Minimum Down Time (h)", 0.0)
                startup_cost_per_mw = zd.char_val(tech, "Start-Up Cost (EUR)", 0.0)
                rows.append(dict(
                    gen=gid, zone=z, tech=tech, category=category, h2_fuel=h2_fuel,
                    mc=_marginal_cost(zd, tech, h2_fuel, net.co2_price, cfg),
                    eff=eff,
                    units=max_units, pmin_unit=pmin_unit, pmax_unit=pmax_unit,
                    ramp_up=ramp_pu * units * cfg.ramp_scale,
                    ramp_dn=ramp_dn * units * cfg.ramp_scale,
                    mustrun_pct=mustrun_pct, pmin_floor=pmin_floor, pmax=cap,
                    msl_frac=msp,
                    min_up_h=min_up_h, min_down_h=min_down_h,
                    startup_cost_eur=startup_cost_per_mw * cap,
                ))
                upper[gid] = np.full(H, cap, dtype=float)

            elif category == dl.CAT_VRES:
                if cap <= 0:
                    continue
                col = dl.VRES_PROFILE.get(tech)
                if col is None or col not in zd.profiles:
                    continue
                cf = _num(zd.profiles[col].to_numpy())
                avail = np.clip(cf, 0.0, None) * cap
                if avail.max() <= 0:
                    continue
                rows.append(dict(gen=gid, zone=z, tech=tech, category=category,
                                 h2_fuel=False, mc=0.0, eff=1.0, pmax=cap))
                upper[gid] = avail

            elif category == dl.CAT_ROR:
                col = "River Flow Energy"
                if col not in zd.profiles:
                    continue
                inflow = np.clip(_num(zd.profiles[col].to_numpy()), 0.0, None)
                avail = np.minimum(inflow, cap) if cap > 0 else inflow
                if avail.max() <= 0:
                    continue
                rows.append(dict(gen=gid, zone=z, tech=tech, category=category,
                                 h2_fuel=False, mc=0.0, eff=1.0, pmax=float(avail.max())))
                upper[gid] = avail

            elif category == dl.CAT_PROFILE:
                col = dl.profile_gen_column(tech)
                if col not in zd.profiles:
                    continue
                avail = np.clip(_num(zd.profiles[col].to_numpy()), 0.0, None)
                if avail.max() <= 0:
                    continue
                hours_limit = (zd.char_val(tech, "Number of Hours (h)", float("inf"))
                              if tech.startswith("DSR") else float("inf"))
                rows.append(dict(
                    gen=gid, zone=z, tech=tech, category=category, h2_fuel=False,
                    mc=_marginal_cost(zd, tech, False, net.co2_price, cfg),
                    eff=1.0, pmax=float(avail.max()), daily_hours_limit=hours_limit))
                upper[gid] = avail

    if rows:
        gens = pd.DataFrame(rows).set_index("gen")
    else:
        gens = pd.DataFrame(
            columns=["zone", "tech", "category", "h2_fuel", "mc", "eff", "pmax"]
        ).set_index(pd.Index([], name="gen"))
    return gens, upper


def _build_storage(zdata: dict[str, ZoneData], cfg: RunConfig):
    rows: list[dict] = []
    inflow: dict[str, np.ndarray] = {}
    H = len(zdata[cfg.zones[0]].profiles)
    zero = np.zeros(H)

    from . import marginal_price_loader as mpl
    h0, h1 = cfg.hour_slice()
    ghours = pd.RangeIndex(h0, h1)
    plexos_inflow = {
        "Hydro reservoir": mpl.load_zone_series(cfg.zones, ghours, mpl.DEFAULT_HYDRO_RESERVOIR_DB),
        "Hydro pondage": mpl.load_zone_series(cfg.zones, ghours, mpl.DEFAULT_HYDRO_PONDAGE_DB),
        "Hydro open_ps": mpl.load_zone_series(cfg.zones, ghours, mpl.DEFAULT_HYDRO_OPEN_PS_DB),
    }

    for z in cfg.zones:
        zd = zdata[z]
        cap = zd.capacities
        e = zd.storage_energy
        prof = zd.profiles

        def col(name):
            return np.clip(_num(prof[name].to_numpy()), 0.0, None) if name in prof else zero

        specs = [
            ("Battery",
             zd.char_val("Battery (MWh)", "Net maximum capacity - generation perspective (MW)"),
             zd.char_val("Battery (MWh)", "Net maximum capacity - demand perspective (MW)"),
             e.get("Battery (MWh)", 0.0), zero,
             max(zd.char_val("Battery (MWh)", "Efficiency (%)", 92.0) / 100.0, 0.1), "electricity"),
            ("Hydro reservoir", cap.get("Hydro (reservoir) (MW)", 0.0), 0.0,
             e.get("Hydro (reservoir) (MWh)", 0.0), col("Reservoir Flow Energy"),
             cfg.default_hydro_efficiency, "electricity"),
            ("Hydro pondage", cap.get("Hydro (pondage) (MW)", 0.0), 0.0,
             e.get("Hydro (pondage) (MWh)", 0.0), col("Pondage Flow Energy"),
             cfg.default_hydro_efficiency, "electricity"),
            ("Hydro open_ps", cap.get("Hydro (open_ps_turbine) (MW)", 0.0),
             abs(cap.get("Hydro (open_ps_pump) (MW)", 0.0)),
             e.get("Hydro (open_ps) (MWh)", 0.0), col("Open_PS Flow Energy"),
             cfg.default_pump_efficiency, "electricity"),
            ("Hydro closed_ps", cap.get("Hydro (closed_ps_turbine) (MW)", 0.0),
             abs(cap.get("Hydro (closed_ps_pump) (MW)", 0.0)),
             e.get("Hydro (closed_ps) (MWh)", 0.0), col("Closed_PS Flow Energy"),
             cfg.default_closed_ps_efficiency, "electricity"),
        ]
        if cfg.enable_h2_storage and not cfg.electricity_only:
            wd = zd.h2_assets.get("Withdraw (Hydrogen) (MW)", 0.0)
            inj = zd.h2_assets.get("Injection (Hydrogen) (MW)", 0.0)
            specs.append(("H2 storage", wd, inj, wd * cfg.h2_storage_hours, zero,
                          cfg.h2_storage_efficiency, "hydrogen"))
        for kind, pdis, pchg, ecap, inf, eff, carrier in specs:
            if ecap <= 0 or pdis <= 0:
                continue
            if kind in plexos_inflow and not np.any(inf):
                inf = np.clip(plexos_inflow[kind][z].to_numpy(), 0.0, None)
            sid = f"{z}|{kind}"
            rows.append(dict(sto=sid, zone=z, kind=kind, pdis=float(pdis),
                             pchg=float(pchg), ecap=float(ecap), eff=float(eff),
                             carrier=carrier))
            inflow[sid] = inf

    storage = pd.DataFrame(rows).set_index("sto") if rows else pd.DataFrame(
        columns=["zone", "kind", "pdis", "pchg", "ecap", "eff", "carrier"]).rename_axis("sto")
    return storage, inflow


def _g_investor_sizing(cfg: RunConfig) -> dict[str, dict]:
    """Per-country electrolyser (rank-assigned against reference H2 load), wind, PV,
    battery, and H2 tank reference capacity for the General Investor, with per-asset
    overrides applied on top."""
    result = _g_investor_sizing_cached(
        str(cfg.zones_db), tuple(cfg.g_investor_electrolyser_capacities_mw),
        cfg.g_investor_renewable_pct_of_electrolyser_mw, cfg.g_investor_wind_to_pv_ratio,
        cfg.g_investor_renewable_capacity_step_mw, cfg.g_investor_electrolyser_efficiency,
        cfg.g_investor_battery_pct_of_electrolyser_mw, cfg.g_investor_battery_duration_hours,
        cfg.g_investor_tank_pct_of_electrolyser_h2, cfg.g_investor_tank_duration_hours,
        cfg.g_investor_battery_tank_step_mw,
    )
    ely_ov = cfg.g_investor_electrolyser_mw_overrides
    wind_ov = cfg.g_investor_wind_mw_overrides
    pv_ov = cfg.g_investor_pv_mw_overrides
    batt_ov = cfg.g_investor_battery_mw_overrides
    tank_ov = cfg.g_investor_tank_mw_overrides
    if ely_ov or wind_ov or pv_ov or batt_ov or tank_ov:
        result = {c: dict(row) for c, row in result.items()}
        for c, row in result.items():
            if c in ely_ov:
                row["electrolyser_mw"] = ely_ov[c]
            if c in wind_ov:
                row["wind_mw"] = wind_ov[c]
            if c in pv_ov:
                row["pv_mw"] = pv_ov[c]
            if c in batt_ov:
                row["battery_mw"] = batt_ov[c]
                row["battery_mwh"] = cfg.g_investor_battery_mwh_overrides.get(
                    c, batt_ov[c] * cfg.g_investor_battery_duration_hours)
            if c in tank_ov:
                row["tank_mw"] = tank_ov[c]
                row["tank_mwh"] = cfg.g_investor_tank_mwh_overrides.get(
                    c, tank_ov[c] * cfg.g_investor_tank_duration_hours)
    return result


@lru_cache(maxsize=8)
def _g_investor_sizing_cached(zones_db: str, capacities: tuple[float, ...],
                               renewable_pct: float, wind_to_pv_ratio: float, renewable_step: float,
                               ely_eff: float, battery_pct: float, battery_hours: float,
                               tank_pct: float, tank_hours: float, batt_tank_step: float) -> dict[str, dict]:
    df = pd.read_parquet(zones_db)
    prof = df[(df["section"] == "profiles") & (df["item"] == "Hydrogen Demand Profile")]
    per_zone_total = prof.groupby("zone")["value_num"].sum()
    n_hours = float(prof.groupby("zone").size().max()) if len(prof) else 8736.0
    country_total = per_zone_total.groupby(lambda z: z[:2]).sum()
    demand_mw = (country_total / n_hours).to_dict()
    demand_mw = {c: v for c, v in demand_mw.items() if v > 0}

    countries_sorted = sorted(demand_mw, key=lambda c: demand_mw[c])
    caps = sorted(capacities)
    n = len(countries_sorted)
    if len(caps) < n:
        caps = caps + [caps[-1]] * (n - len(caps))
    elif len(caps) > n:
        caps = caps[len(caps) - n:]
    cap_by_country = dict(zip(countries_sorted, caps))

    def round_step(v: float, step: float) -> float:
        """Round to the nearest multiple of ``step``, floored at one step (never 0)."""
        return max(step, round(v / step) * step)

    result = {}
    for c, ely in cap_by_country.items():
        combined = renewable_pct * ely
        pv_raw = combined / (1.0 + wind_to_pv_ratio)
        wind_raw = wind_to_pv_ratio * pv_raw
        wind_mw = round_step(wind_raw, renewable_step)
        pv_mw = round_step(pv_raw, renewable_step)

        ely_h2 = ely * ely_eff
        battery_mw = round_step(battery_pct * ely, batt_tank_step)
        tank_mw = round_step(tank_pct * ely_h2, batt_tank_step)

        result[c] = {
            "demand_mw": demand_mw[c], "electrolyser_mw": ely,
            "wind_mw": wind_mw, "pv_mw": pv_mw,
            "battery_mw": battery_mw, "battery_mwh": battery_mw * battery_hours,
            "tank_mw": tank_mw, "tank_mwh": tank_mw * tank_hours,
        }
    return result


@lru_cache(maxsize=4)
def _g_investor_renewable_profile_info(zones_db: str) -> dict[str, tuple[bool, float, bool, float]]:
    """Per zone: ``(has_wind_data, wind_max, has_solar_data, solar_max)`` over the full stored year."""
    df = pd.read_parquet(zones_db)
    prof = df[df["section"] == "profiles"]
    wind = prof[prof["item"] == "Wind_Onshore Profile"].groupby("zone")["value_num"]
    solar = prof[prof["item"] == "Solar Profile"].groupby("zone")["value_num"]
    wind_max = wind.max()
    solar_max = solar.max()
    zones = set(wind_max.index) | set(solar_max.index)
    return {z: (float(wind_max.get(z, 0.0)) > 0.0, float(wind_max.get(z, 0.0)),
               float(solar_max.get(z, 0.0)) > 0.0, float(solar_max.get(z, 0.0)))
           for z in zones}


def _incidence(members: pd.Series, zones: list[str], dim: str) -> xr.DataArray:
    """One-hot (member, zone) matrix from a Series mapping member -> zone."""
    A = np.zeros((len(members), len(zones)))
    zpos = {z: i for i, z in enumerate(zones)}
    for i, z in enumerate(members.to_numpy()):
        A[i, zpos[z]] = 1.0
    return xr.DataArray(A, coords={dim: members.index, ZONE: zones}, dims=[dim, ZONE])


def uc_candidates(gens: pd.DataFrame) -> list[str]:
    """Fleets eligible for cfg.enable_uc: no must-run floor and min_up_h/min_down_h > 1h."""
    return [gid for gid, row in gens.iterrows()
            if row.get("pmin_floor", 0.0) == 0.0
            and max(row.get("min_up_h", 0.0), row.get("min_down_h", 0.0)) > 1.0]


def build_model(zdata: dict[str, ZoneData], net: NetworkData, cfg: RunConfig,
                cyclic: bool | None = None,
                fixed_uc_profile: dict[str, np.ndarray] | None = None) -> BuildResult:
    """Build the dispatch LP (or, with ``cfg.enable_uc``, a small MILP; see
    pipeline.solve_scenario for the two-pass fixed_uc_profile orchestration)."""
    zones = cfg.zones
    H = len(zdata[zones[0]].profiles)
    hours = pd.Index(range(H), name=HOUR)
    zidx = pd.Index(zones, name=ZONE)

    gens, gupper = _build_generators(zdata, net, cfg)
    if cfg.use_plexos_renewable_override:
        gens, gupper = _override_renewable_upper_with_plexos(zdata, gens, gupper, zones, cfg)
    storage, sinflow = _build_storage(zdata, cfg)

    uc_gens = uc_candidates(gens) if cfg.enable_uc else []

    m = linopy.Model()

    gen_index = gens.index
    upper_mat = np.vstack([gupper[g] for g in gen_index]) if len(gen_index) > 0 else np.zeros((0, H))
    gen_upper = xr.DataArray(upper_mat, coords={GEN: gen_index, HOUR: hours}, dims=[GEN, HOUR])
    floor_vec = np.nan_to_num(gens["pmin_floor"].to_numpy(float)) \
        if "pmin_floor" in gens.columns else np.zeros(len(gen_index))
    gen_lower = xr.DataArray(
        np.tile(floor_vec[:, None], (1, H)), coords={GEN: gen_index, HOUR: hours}, dims=[GEN, HOUR]
    )
    if fixed_uc_profile is not None:
        gen_upper = gen_upper.copy()
        for gid, prof in fixed_uc_profile.items():
            msl_frac = float(gens.loc[gid, "msl_frac"])
            gen_upper.loc[{GEN: gid}] = prof
            gen_lower.loc[{GEN: gid}] = msl_frac * prof
    gen_p = m.add_variables(lower=gen_lower, upper=gen_upper, name="gen_p")

    if cfg.use_plexos_renewable_override:
        _joint_renewable_constraints(m, gens, gen_p, zones, hours, cfg)

    A_gen = _incidence(gens["zone"], zones, GEN)
    gen_by_zone = (A_gen * gen_p).sum(GEN)

    uc_x_on = None
    uc_startup_obj = 0.0
    if fixed_uc_profile is None and uc_gens:
        uc_idx = pd.Index(uc_gens, name=GEN)
        uc_x_on = m.add_variables(binary=True, coords=[uc_idx, hours], name="uc_on")
        uc_y_start = m.add_variables(lower=0.0, upper=1.0, coords=[uc_idx, hours], name="uc_start")
        uc_z_stop = m.add_variables(lower=0.0, upper=1.0, coords=[uc_idx, hours], name="uc_stop")

        uc_cap = xr.DataArray(gens.loc[uc_gens, "pmax"].to_numpy(float), coords={GEN: uc_idx}, dims=[GEN])
        uc_msl = xr.DataArray(gens.loc[uc_gens, "msl_frac"].to_numpy(float), coords={GEN: uc_idx}, dims=[GEN])
        m.add_constraints(gen_p.sel({GEN: uc_idx}) <= uc_cap * uc_x_on, name="uc_cap_link")
        m.add_constraints(gen_p.sel({GEN: uc_idx}) >= uc_msl * uc_cap * uc_x_on, name="uc_msl_link")

        delta_x = uc_x_on - uc_x_on.shift({HOUR: 1}, fill_value=0.0)
        m.add_constraints(delta_x == uc_y_start - uc_z_stop, name="uc_startstop")

        for gid in uc_gens:
            min_on = max(int(round(gens.loc[gid, "min_up_h"])), 1)
            min_off = max(int(round(gens.loc[gid, "min_down_h"])), 1)
            y_g = uc_y_start.sel({GEN: gid})
            z_g = uc_z_stop.sel({GEN: gid})
            x_g = uc_x_on.sel({GEN: gid})
            roll_y = sum(y_g.shift({HOUR: k}, fill_value=0.0) for k in range(min_on))
            roll_z = sum(z_g.shift({HOUR: k}, fill_value=0.0) for k in range(min_off))
            m.add_constraints(roll_y <= x_g, name=f"uc_minup_{gid}")
            m.add_constraints(roll_z <= 1 - x_g, name=f"uc_mindown_{gid}")

        uc_startup_cost = xr.DataArray(
            gens.loc[uc_gens, "startup_cost_eur"].to_numpy(float), coords={GEN: uc_idx}, dims=[GEN]
        )
        uc_startup_obj = (uc_startup_cost * uc_y_start).sum()

    commit = gens[gens["category"] == dl.CAT_COMMIT].copy()

    if "daily_hours_limit" in gens.columns:
        capped = gens[np.isfinite(gens["daily_hours_limit"].to_numpy(float))]
        if len(capped) > 0:
            cap_idx = capped.index
            day_budget = xr.DataArray(
                capped["pmax"].to_numpy(float) * capped["daily_hours_limit"].to_numpy(float),
                coords={GEN: cap_idx}, dims=[GEN],
            )
            gp_capped = gen_p.sel({GEN: cap_idx})
            n_days = len(hours) // 24
            for d in range(n_days):
                day_hours = hours[d * 24:(d + 1) * 24]
                window = gp_capped.sel({HOUR: day_hours}).sum(HOUR)
                m.add_constraints(window <= day_budget, name=f"daily_hours_cap_{d}")

    have_sto = len(storage) > 0
    if have_sto:
        sidx = storage.index
        pdis = xr.DataArray(storage["pdis"].to_numpy(float), coords={STO: sidx}, dims=[STO])
        pchg = xr.DataArray(storage["pchg"].to_numpy(float), coords={STO: sidx}, dims=[STO])
        ecap = xr.DataArray(storage["ecap"].to_numpy(float), coords={STO: sidx}, dims=[STO])
        eff = storage["eff"].to_numpy(float)
        dis = m.add_variables(lower=0.0, upper=_bc(pdis, hours), name="dis")
        ch = m.add_variables(lower=0.0, upper=_bc(pchg, hours), name="ch")
        soc = m.add_variables(lower=0.0, upper=_bc(ecap, hours), name="soc")
        spill = m.add_variables(lower=0.0, name="spill", coords=[sidx, hours])

        A_sto = _incidence(storage["zone"], zones, STO)
        carr = storage["carrier"].to_numpy()
        mask_e = xr.DataArray((carr == "electricity").astype(float), coords={STO: sidx}, dims=[STO])
        mask_h = xr.DataArray((carr == "hydrogen").astype(float), coords={STO: sidx}, dims=[STO])
        dis_by_zone = (A_sto * mask_e * dis).sum(STO)
        ch_by_zone = (A_sto * mask_e * ch).sum(STO)
        dis_h2_by_zone = (A_sto * mask_h * dis).sum(STO)
        ch_h2_by_zone = (A_sto * mask_h * ch).sum(STO)

        soc0 = cfg.initial_soc_fraction * storage["ecap"].to_numpy(float)
        inflow_mat = np.vstack([sinflow[s] for s in sidx])
        eff_da = xr.DataArray(eff, coords={STO: sidx}, dims=[STO])
        rhs_mat = inflow_mat.copy()
        rhs_mat[:, 0] = rhs_mat[:, 0] + soc0
        rhs = xr.DataArray(rhs_mat, coords={STO: sidx, HOUR: hours}, dims=[STO, HOUR])
        m.add_constraints(soc - soc.shift({HOUR: 1}) - eff_da * ch + dis + spill == rhs,
                          name="soc_balance")
        if cfg.cyclic_storage if cyclic is None else cyclic:
            end = xr.DataArray(soc0, coords={STO: sidx}, dims=[STO])
            m.add_constraints(soc.sel({HOUR: H - 1}) >= end, name="soc_cyclic")
    else:
        dis_by_zone = ch_by_zone = 0.0
        dis_h2_by_zone = ch_h2_by_zone = 0.0

    ely_cap = np.array([zdata[z].capacities.get("Electrolyser (MW)", 0.0) for z in zones])
    ely_eff = np.array([max(zdata[z].char_val("Electrolyser (MW)", "Efficiency (%)", 68.0) / 100.0, 1e-6)
                        for z in zones])
    ely_p = m.add_variables(lower=0.0, upper=_bc_z(ely_cap, zidx, hours), name="ely_p")

    if not cfg.electricity_only:
        ely_eff_da = xr.DataArray(ely_eff, coords={ZONE: zidx}, dims=[ZONE])
        ely_h2_term = ely_eff_da * ely_p
        term_cap = np.array([zdata[z].h2_assets.get("Terminal (Hydrogen) (MW)", 0.0) for z in zones])
        term_h2 = m.add_variables(lower=0.0, upper=_bc_z(term_cap, zidx, hours), name="term_h2")

        A_h2 = A_gen.copy()
        h2_coeff = np.where(gens["h2_fuel"].to_numpy(), 1.0 / gens["eff"].to_numpy(), 0.0)
        A_h2 = A_h2 * xr.DataArray(h2_coeff, coords={GEN: gen_index}, dims=[GEN])
        h2_cons_by_zone = (A_h2 * gen_p).sum(GEN)
    else:
        term_cap = np.zeros(len(zones))

    net_e, fe_pos, fe_neg = _flow_terms(m, net.elec, zones, hours, "e")
    net_h, fh_pos, fh_neg = (0.0, None, None) if cfg.electricity_only \
        else _flow_terms(m, net.hydrogen, zones, hours, "h")

    demand_e = _profile_da(zdata, zones, hours, "Electricity Demand Profile")
    demand_h = _profile_da(zdata, zones, hours, "Hydrogen Demand Profile") if not cfg.electricity_only else None
    if cfg.subtract_dsr_implicit:
        from . import marginal_price_loader as mpl
        h0, h1 = cfg.hour_slice()
        ghours = pd.RangeIndex(h0, h1)
        dsr_df = mpl.load_zone_series(zones, ghours, mpl.DEFAULT_DSR_IMPLICIT_DB)
        dsr_da = xr.DataArray(dsr_df.to_numpy().T, coords={ZONE: zidx, HOUR: hours}, dims=[ZONE, HOUR])
        demand_e = demand_e - dsr_da

    external_e, ext_e_obj = _priced_external_elec(m, zones, hours, cfg)
    smr_gen = None
    smr_obj = 0.0
    ext_h2_obj = 0.0
    if not cfg.electricity_only:
        cross_border_h2, ext_h2_obj = _priced_external_h2(m, zones, hours, cfg)
        external_h2 = cross_border_h2 + _fixed_h2_supply_injection(zones, hours, cfg)
        smr_gen, smr_obj, smr_fixed_da = _smr_priced_generation(m, zones, hours, cfg)
        external_h2 = external_h2 - smr_fixed_da

    shed_e = m.add_variables(lower=0.0, coords=[zidx, hours], name="shed_e")
    dump_e = m.add_variables(lower=0.0, coords=[zidx, hours], name="dump_e")
    if not cfg.electricity_only:
        shed_h = m.add_variables(lower=0.0, coords=[zidx, hours], name="shed_h")
        dump_h = m.add_variables(lower=0.0, coords=[zidx, hours], name="dump_h")

    elec_lhs = (gen_by_zone + dis_by_zone - ch_by_zone - ely_p + net_e
                + external_e + shed_e - dump_e)
    m.add_constraints(elec_lhs == demand_e, name="elec_balance")

    if not cfg.electricity_only:
        h2_lhs = (ely_h2_term + term_h2 + net_h + external_h2
                  + (smr_gen if smr_gen is not None else 0.0)
                  + dis_h2_by_zone - ch_h2_by_zone + shed_h
                  - h2_cons_by_zone - dump_h)
        m.add_constraints(h2_lhs == demand_h, name="h2_balance")

    ramp_commit = commit.drop(index=uc_gens, errors="ignore") if uc_gens else commit
    ramp_cidx = ramp_commit.index
    if len(ramp_commit) > 0:
        rup = xr.DataArray(ramp_commit["ramp_up"].to_numpy(float), coords={GEN: ramp_cidx}, dims=[GEN])
        rdn = xr.DataArray(ramp_commit["ramp_dn"].to_numpy(float), coords={GEN: ramp_cidx}, dims=[GEN])
        gp_c = gen_p.sel({GEN: ramp_cidx})
        delta = (gp_c - gp_c.shift({HOUR: 1})).isel({HOUR: slice(1, None)})
        if ramp_commit["ramp_up"].to_numpy(float).max() > 0:
            m.add_constraints(delta <= rup, name="ramp_up")
        if ramp_commit["ramp_dn"].to_numpy(float).max() > 0:
            m.add_constraints(-delta <= rdn, name="ramp_dn")

    mc = xr.DataArray(gens["mc"].to_numpy(float), coords={GEN: gen_index}, dims=[GEN])
    obj = (mc * gen_p).sum() \
        + cfg.voll_eur_per_mwh * shed_e.sum() \
        + cfg.dump_penalty_eur_per_mwh * dump_e.sum() \
        + uc_startup_obj + ext_e_obj
    if not cfg.electricity_only:
        obj = obj + cfg.h2_terminal_price * term_h2.sum() \
            + cfg.voll_eur_per_mwh * shed_h.sum() \
            + cfg.dump_penalty_eur_per_mwh * dump_h.sum() \
            + ext_h2_obj + smr_obj
    if have_sto:
        obj = obj + cfg.storage_op_cost_eur_per_mwh * (ch.sum() + dis.sum())
    m.add_objective(obj)

    br = BuildResult(m, cfg, zones, hours, gens, commit, storage, gen_upper,
                     demand_e, demand_h, external_e, external_h2, net.elec, net.hydrogen, net,
                     uc_gens=(uc_gens if uc_x_on is not None else None))
    br._ely_eff = pd.Series(ely_eff, index=zones)
    br._ely_cap = pd.Series(ely_cap, index=zones)
    br._term_cap = pd.Series(term_cap, index=zones)
    return br


def marginal_prices(build: BuildResult):
    """Zonal marginal prices (EUR/MWh) as the duals of the nodal balances; returns (price_e, price_h)."""
    price_e = build.model.constraints["elec_balance"].dual
    price_h = build.model.constraints["h2_balance"].dual if not build.cfg.electricity_only else None
    return price_e, price_h


def uc_fixed_profile_and_cost(build: BuildResult) -> tuple[dict[str, np.ndarray], float]:
    """From a solved pass-1 MILP build: the solved 0/pmax commitment profile per gen
    (for pass 2's fixed_uc_profile) and the total start-up cost incurred."""
    x_on_sol = build.model.solution["uc_on"]
    y_start_sol = build.model.solution["uc_start"]
    fixed_profile: dict[str, np.ndarray] = {}
    total_cost = 0.0
    for gid in build.uc_gens:
        cap = float(build.gens.loc[gid, "pmax"])
        onoff = np.round(x_on_sol.sel({GEN: gid}).to_numpy())
        fixed_profile[gid] = onoff * cap
        starts = y_start_sol.sel({GEN: gid}).to_numpy().sum()
        total_cost += starts * float(build.gens.loc[gid, "startup_cost_eur"])
    return fixed_profile, total_cost


def _bc(da_over_sto: xr.DataArray, hours: pd.Index) -> xr.DataArray:
    """Broadcast a per-storage DataArray to (sto, hour)."""
    return da_over_sto.expand_dims({HOUR: hours}).transpose(STO, HOUR)


def _bc_z(values: np.ndarray, zidx: pd.Index, hours: pd.Index) -> xr.DataArray:
    da = xr.DataArray(values, coords={ZONE: zidx}, dims=[ZONE])
    return da.expand_dims({HOUR: hours}).transpose(ZONE, HOUR)


def _profile_da(zdata, zones, hours, col) -> xr.DataArray:
    mat = np.vstack([_num(zdata[z].profiles[col].to_numpy()) if col in zdata[z].profiles
                     else np.zeros(len(hours)) for z in zones])
    return xr.DataArray(mat, coords={ZONE: pd.Index(zones, name=ZONE), HOUR: hours},
                        dims=[ZONE, HOUR])


def _h2_main_zones(cfg: RunConfig) -> dict[str, str]:
    """Main H2 zone per country = the country's zone with the most H2 demand, computed over
    the full declared zone universe so it's stable regardless of the current run's selection."""
    return _h2_main_zones_cached(str(cfg.zones_db))


@lru_cache(maxsize=4)
def _h2_main_zones_cached(zones_db: str) -> dict[str, str]:
    from .config import discover_zones
    all_zones = discover_zones(zones_db)
    df = pd.read_parquet(zones_db)
    prof = df[(df["section"] == "profiles") & (df["item"] == "Hydrogen Demand Profile")]
    dem = prof.groupby("zone")["value_num"].sum()
    best: dict[str, str] = {}
    best_val: dict[str, float] = {}
    for z in all_zones:
        d = float(dem.get(z, 0.0))
        c = z[:2]
        if c not in best or d > best_val[c]:
            best[c], best_val[c] = z, d
    return best


def _priced_external_elec(m: linopy.Model, zones: list[str], hours: pd.Index, cfg: RunConfig):
    """Priced/controllable import & export legs for every zone's external electricity
    neighbours, capped at real line capacity and priced at the neighbour's PLEXOS marginal
    price; returns (net_injection_expr, objective_cost_expr)."""
    from . import marginal_price_loader as mpl

    zidx = pd.Index(zones, name=ZONE)
    edf = pd.read_parquet(Path(cfg.exports_dir) / "crossborder_electricity_2030.parquet")
    legs = exports_loader.elec_border_legs(zones, edf)

    if not legs:
        zero = xr.DataArray(np.zeros((len(zones), len(hours))),
                            coords={ZONE: zidx, HOUR: hours}, dims=[ZONE, HOUR])
        return zero, 0.0

    h0, h1 = cfg.hour_slice()
    pairs = sorted(legs)
    pname = "extleg"
    pidx = pd.Index([f"{z}|{n}" for z, n in pairs], name=pname)

    line_caps = nl.border_line_caps("electricity", cfg.networks_db)

    def _border_cap(z: str, n: str) -> tuple[float, float]:
        """(import cap z<-n, export cap z->n) MW; 0 if no line exists."""
        if (z, n) in line_caps:
            ft, tf = line_caps[(z, n)]
            return tf, ft
        if (n, z) in line_caps:
            ft, tf = line_caps[(n, z)]
            return ft, tf
        return 0.0, 0.0

    imp_vec = np.array([_border_cap(z, n)[0] for z, n in pairs])
    exp_vec = np.array([_border_cap(z, n)[1] for z, n in pairs])
    imp_cap = np.tile(imp_vec[:, None], (1, len(hours)))
    exp_cap = np.tile(exp_vec[:, None], (1, len(hours)))

    if cfg.external_import_leg_cap:
        for i, (z, n) in enumerate(pairs):
            leg_cap = cfg.external_import_leg_cap.get(f"{z}|{n}")
            if leg_cap is not None:
                imp_cap[i, :] = np.minimum(imp_cap[i, :], np.asarray(leg_cap, dtype=float))
    if cfg.external_export_leg_cap:
        for i, (z, n) in enumerate(pairs):
            leg_cap = cfg.external_export_leg_cap.get(f"{z}|{n}")
            if leg_cap is not None:
                exp_cap[i, :] = np.minimum(exp_cap[i, :], np.asarray(leg_cap, dtype=float))

    neighbors = sorted({n for _, n in pairs})
    ghours = pd.RangeIndex(h0, h1)
    price_df = mpl.load_zone_series(neighbors, ghours, mpl.DEFAULT_MARGINAL_PRICE_ELEC_DB)
    price_mat = np.vstack([price_df[n].to_numpy() for _, n in pairs])

    imp_cap_da = xr.DataArray(imp_cap, coords={pname: pidx, HOUR: hours}, dims=[pname, HOUR])
    exp_cap_da = xr.DataArray(exp_cap, coords={pname: pidx, HOUR: hours}, dims=[pname, HOUR])
    price_da = xr.DataArray(price_mat, coords={pname: pidx, HOUR: hours}, dims=[pname, HOUR])

    imp = m.add_variables(lower=0.0, upper=imp_cap_da, name="ext_imp")
    exp = m.add_variables(lower=0.0, upper=exp_cap_da, name="ext_exp")

    A = np.zeros((len(pairs), len(zones)))
    zpos = {z: i for i, z in enumerate(zones)}
    for i, (z, _n) in enumerate(pairs):
        A[i, zpos[z]] = 1.0
    A_da = xr.DataArray(A, coords={pname: pidx, ZONE: zidx}, dims=[pname, ZONE])

    net_injection = (A_da * imp).sum(pname) - (A_da * exp).sum(pname)
    obj = (price_da * imp).sum() - (price_da * exp).sum()
    return net_injection, obj


def _priced_external_h2(m: linopy.Model, zones: list[str], hours: pd.Index, cfg: RunConfig):
    """Hydrogen analogue of ``_priced_external_elec``: priced/controllable cross-border H2
    legs from each country's main H2 zone, capped at real pipeline capacity. Excludes SMR
    and virtual-source H2 (see ``_fixed_h2_supply_injection``)."""
    from . import marginal_price_loader as mpl

    zidx = pd.Index(zones, name=ZONE)
    main_map = _h2_main_zones(cfg)
    hdf = pd.read_parquet(Path(cfg.exports_dir) / "crossborder_hydrogen_2030.parquet")
    legs = exports_loader.h2_border_legs(zones, main_map, hdf)

    if not legs:
        zero = xr.DataArray(np.zeros((len(zones), len(hours))),
                            coords={ZONE: zidx, HOUR: hours}, dims=[ZONE, HOUR])
        return zero, 0.0

    h0, h1 = cfg.hour_slice()
    pairs = sorted(legs)
    pname = "h2leg"
    pidx = pd.Index([f"{z}|{n}" for z, n in pairs], name=pname)

    line_caps = nl.border_line_caps("hydrogen", cfg.networks_db)
    sel_countries = {z[:2] for z in zones}
    country_node: dict[str, str] = {c: z for c, z in main_map.items() if c not in sel_countries}
    for pair in line_caps:
        for n in pair:
            c = n[:2]
            if c not in sel_countries and c not in country_node:
                country_node[c] = n

    def _border_cap(z: str, n: str) -> tuple[float, float]:
        """(import cap z<-n, export cap z->n) MW; 0 if no line/no mapped node."""
        node = country_node.get(n)
        if node is None:
            return 0.0, 0.0
        if (z, node) in line_caps:
            ft, tf = line_caps[(z, node)]
            return tf, ft
        if (node, z) in line_caps:
            ft, tf = line_caps[(node, z)]
            return ft, tf
        return 0.0, 0.0

    imp_vec = np.array([_border_cap(z, n)[0] for z, n in pairs])
    exp_vec = np.array([_border_cap(z, n)[1] for z, n in pairs])
    imp_cap = np.tile(imp_vec[:, None], (1, len(hours)))
    exp_cap = np.tile(exp_vec[:, None], (1, len(hours)))

    neighbors = sorted({n for _, n in pairs})
    ghours = pd.RangeIndex(h0, h1)
    price_df = mpl.load_zone_series([f"{n}_H2" for n in neighbors], ghours,
                                    mpl.DEFAULT_MARGINAL_PRICE_H2_DB)
    price_mat = np.vstack([price_df[f"{n}_H2"].to_numpy() for _, n in pairs])

    imp_cap_da = xr.DataArray(imp_cap, coords={pname: pidx, HOUR: hours}, dims=[pname, HOUR])
    exp_cap_da = xr.DataArray(exp_cap, coords={pname: pidx, HOUR: hours}, dims=[pname, HOUR])
    price_da = xr.DataArray(price_mat, coords={pname: pidx, HOUR: hours}, dims=[pname, HOUR])

    imp = m.add_variables(lower=0.0, upper=imp_cap_da, name="h2_ext_imp")
    exp = m.add_variables(lower=0.0, upper=exp_cap_da, name="h2_ext_exp")

    A = np.zeros((len(pairs), len(zones)))
    zpos = {z: i for i, z in enumerate(zones)}
    for i, (z, _n) in enumerate(pairs):
        A[i, zpos[z]] = 1.0
    A_da = xr.DataArray(A, coords={pname: pidx, ZONE: zidx}, dims=[pname, ZONE])

    net_injection = (A_da * imp).sum(pname) - (A_da * exp).sum(pname)
    obj = (price_da * imp).sum() - (price_da * exp).sum()
    return net_injection, obj


def _smr_priced_generation(m: linopy.Model, zones: list[str], hours: pd.Index, cfg: RunConfig):
    """Steam-Methane-Reformer as a real generation variable, bounded above by PLEXOS's realized
    hourly SMR output per country and priced at PLEXOS's realized H2 marginal price; returns
    (smr_gen variable, objective cost expr, fixed-injection DataArray to subtract elsewhere)."""
    from . import marginal_price_loader as mpl

    zidx = pd.Index(zones, name=ZONE)
    main_map = _h2_main_zones(cfg)
    h0, h1 = cfg.hour_slice()
    ghours = pd.RangeIndex(h0, h1)
    smr_df = pd.read_parquet(Path(cfg.exports_dir) / "smr_production_2030.parquet")

    fixed_inj = exports_loader.smr_injection(zones, main_map, smr_df)
    fixed_rows = [fixed_inj.get(z, np.zeros(len(smr_df)))[h0:h1] for z in zones]
    fixed_da = xr.DataArray(np.vstack(fixed_rows), coords={ZONE: zidx, HOUR: hours}, dims=[ZONE, HOUR])

    countries_here = sorted({c for c, z in main_map.items() if z in zones})
    price_df = mpl.load_zone_series([f"{c}_H2" for c in countries_here], ghours,
                                    mpl.DEFAULT_MARGINAL_PRICE_H2_DB)
    zpos = {z: i for i, z in enumerate(zones)}
    upper = np.zeros((len(zones), len(hours)))
    cost = np.zeros((len(zones), len(hours)))
    for c in countries_here:
        i = zpos[main_map[c]]
        if c in smr_df.columns:
            vals = np.clip(pd.to_numeric(smr_df[c], errors="coerce").fillna(0.0).to_numpy()[h0:h1], 0.0, None)
            has_smr = vals > 0
            upper[i] = np.where(has_smr, vals + 1e-3, 0.0)
            cost[i] = np.where(has_smr, price_df[f"{c}_H2"].to_numpy(), 0.0)

    upper_da = xr.DataArray(upper, coords={ZONE: zidx, HOUR: hours}, dims=[ZONE, HOUR])
    cost_da = xr.DataArray(cost, coords={ZONE: zidx, HOUR: hours}, dims=[ZONE, HOUR])
    smr_gen = m.add_variables(lower=0.0, upper=upper_da, name="smr_gen")
    obj = (cost_da * smr_gen).sum()
    return smr_gen, obj, fixed_da


def _fixed_h2_supply_injection(zones: list[str], hours: pd.Index, cfg: RunConfig) -> xr.DataArray:
    """SMR output plus virtual-source H2 as a fixed/uncapacitated (zone, hour) injection
    DataArray, added back on top of ``_priced_external_h2``'s cross-border legs."""
    main_map = _h2_main_zones(cfg)
    h0, h1 = cfg.hour_slice()
    smr = pd.read_parquet(Path(cfg.exports_dir) / "smr_production_2030.parquet")
    hdf = pd.read_parquet(Path(cfg.exports_dir) / "crossborder_hydrogen_2030.parquet")
    inj = exports_loader.smr_injection(zones, main_map, smr)
    for M, arr in exports_loader.exogenous_h2_injection(zones, main_map, hdf).items():
        inj[M] = inj.get(M, np.zeros(len(hdf))) + arr
    zidx = pd.Index(zones, name=ZONE)
    rows = [inj.get(z, np.zeros(len(hdf)))[h0:h1] for z in zones]
    return xr.DataArray(np.vstack(rows), coords={ZONE: zidx, HOUR: hours}, dims=[ZONE, HOUR])


_RENEWABLE_SINGLE = [
    ("wind_onshore", "Wind (onshore) (MW)"),
    ("wind_offshore", "Wind (offshore) (MW)"),
    ("ror", "Hydro (river) (MW)"),
]
_RENEWABLE_JOINT = [
    ("solar_pv", ["Solar (MW)", "Solar (rooftop) (MW)"]),
    ("solar_thermal", ["Solar (thermal) (MW)", "Solar (thermal_with_storage) (MW)"]),
    ("other_res", ["Other RES (biomass) (MW)", "Other RES (geothermal) (MW)",
                   "Other RES (marine) (MW)", "Other RES (waste) (MW)", "Other RES (unknown) (MW)"]),
]


def _renewable_plexos_dbs():
    from . import marginal_price_loader as mpl
    return {
        "wind_onshore": mpl.DEFAULT_WIND_ONSHORE_DB, "wind_offshore": mpl.DEFAULT_WIND_OFFSHORE_DB,
        "ror": mpl.DEFAULT_ROR_DB, "solar_pv": mpl.DEFAULT_SOLAR_PV_DB,
        "solar_thermal": mpl.DEFAULT_SOLAR_THERMAL_DB, "other_res": mpl.DEFAULT_OTHER_RES_DB,
    }


def _new_renewable_row(z: str, tech: str, category: str, pmax: float) -> dict:
    """Minimal gens row for a renewable generator created purely from PLEXOS data."""
    return dict(gen=f"{z}|{tech}", zone=z, tech=tech, category=category,
               h2_fuel=False, mc=0.0, eff=1.0, pmax=pmax)


def _override_renewable_upper_with_plexos(zdata: dict[str, ZoneData], gens: pd.DataFrame,
                                          gupper: dict[str, np.ndarray], zones: list[str],
                                          cfg: RunConfig) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Replace every renewable generator's hourly availability with PLEXOS's realized
    generation for that technology, creating a generator for any zone with real capacity
    but an all-zero profile. Must run before gen_p's upper bound DataArray is built."""
    from . import marginal_price_loader as mpl
    dbs = _renewable_plexos_dbs()
    h0, h1 = cfg.hour_slice()
    ghours = pd.RangeIndex(h0, h1)
    gidx = set(gens.index)
    new_rows: list[dict] = []

    def _cap(z: str, tech: str) -> float:
        return float(zdata[z].capacities.get(tech, 0.0) or 0.0)

    for key, tech in _RENEWABLE_SINGLE:
        category = dl.CAT_ROR if key == "ror" else dl.CAT_VRES
        px = mpl.load_zone_series(zones, ghours, dbs[key])
        for z in zones:
            gid = f"{z}|{tech}"
            vals = px[z].to_numpy()
            if gid in gidx:
                gupper[gid] = vals
            elif _cap(z, tech) > 0:
                new_rows.append(_new_renewable_row(z, tech, category, float(vals.max())))
                gupper[gid] = vals
                gidx.add(gid)

    for key, techs in _RENEWABLE_JOINT:
        px = mpl.load_zone_series(zones, ghours, dbs[key])
        for z in zones:
            vals = px[z].to_numpy()
            for t in techs:
                gid = f"{z}|{t}"
                if gid in gidx:
                    gupper[gid] = vals
                elif _cap(z, t) > 0:
                    new_rows.append(_new_renewable_row(z, t, dl.CAT_VRES, float(vals.max())))
                    gupper[gid] = vals
                    gidx.add(gid)

    if new_rows:
        gens = pd.concat([gens, pd.DataFrame(new_rows).set_index("gen")])
    return gens, gupper


def _joint_renewable_constraints(m: linopy.Model, gens: pd.DataFrame, gen_p, zones: list[str],
                                 hours: pd.Index, cfg: RunConfig) -> None:
    """For _RENEWABLE_JOINT techs, cap sum(gen_p over the group) at PLEXOS's realized total."""
    from . import marginal_price_loader as mpl
    dbs = _renewable_plexos_dbs()
    h0, h1 = cfg.hour_slice()
    ghours = pd.RangeIndex(h0, h1)
    gidx = set(gens.index)

    for key, techs in _RENEWABLE_JOINT:
        px = mpl.load_zone_series(zones, ghours, dbs[key])
        for z in zones:
            present = [f"{z}|{t}" for t in techs if f"{z}|{t}" in gidx]
            if not present:
                continue
            cap_da = xr.DataArray(px[z].to_numpy(), coords={HOUR: hours}, dims=[HOUR])
            expr = sum(gen_p.sel({GEN: gid}) for gid in present)
            m.add_constraints(expr <= cap_da, name=f"plexos_cap_{key}_{z}")


def _flow_terms(m: linopy.Model, lines: list[Line], zones: list[str], hours: pd.Index, tag: str):
    """Create directional flow vars and return the per-zone net-import expression."""
    if not lines:
        return 0.0, None, None
    lidx = pd.Index([f"{tag}{i}:{l.frm}->{l.to}" for i, l in enumerate(lines)], name=f"line_{tag}")
    cap_ft = xr.DataArray([l.cap_ft for l in lines], coords={lidx.name: lidx}, dims=[lidx.name])
    cap_tf = xr.DataArray([l.cap_tf for l in lines], coords={lidx.name: lidx}, dims=[lidx.name])
    loss = np.array([l.loss for l in lines])

    fpos = m.add_variables(lower=0.0, upper=_bc_line(cap_ft, hours), name=f"f{tag}_pos")
    fneg = m.add_variables(lower=0.0, upper=_bc_line(cap_tf, hours), name=f"f{tag}_neg")

    Cfrom = np.zeros((len(lines), len(zones)))
    Cto = np.zeros((len(lines), len(zones)))
    zpos = {z: i for i, z in enumerate(zones)}
    for i, l in enumerate(lines):
        Cfrom[i, zpos[l.frm]] = 1.0
        Cto[i, zpos[l.to]] = 1.0
    dim = lidx.name
    Cfrom = xr.DataArray(Cfrom, coords={dim: lidx, ZONE: zones}, dims=[dim, ZONE])
    Cto = xr.DataArray(Cto, coords={dim: lidx, ZONE: zones}, dims=[dim, ZONE])
    lloss = xr.DataArray(loss, coords={dim: lidx}, dims=[dim])

    coeff_pos = Cto * (1 - lloss) - Cfrom
    coeff_neg = Cfrom * (1 - lloss) - Cto
    net_import = (coeff_pos * fpos).sum(dim) + (coeff_neg * fneg).sum(dim)
    return net_import, fpos, fneg


def _bc_line(da: xr.DataArray, hours: pd.Index) -> xr.DataArray:
    dim = da.dims[0]
    return da.expand_dims({HOUR: hours}).transpose(dim, HOUR)
