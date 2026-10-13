"""Standalone General Investor optimization for one zone/day-range, priced by the trained price proxy."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import linopy
import linopy.solvers as linopy_solvers
from economic_dispatch.config import RunConfig, discover_zones
from economic_dispatch import model as ed_model
from economic_dispatch import data_loader as ed_dl

from price_model.multivariate import predict as model_predict
from price_model.neighbors import add_neighbor_features, add_candidate_neighbor_prices, load_adjacency
from price_model import api as price_api

HOURS_PER_DAY = 24
TOTAL_YEAR_DAYS = 364
N_MONTHS = 12
TOTAL_YEAR_HOURS = TOTAL_YEAR_DAYS * HOURS_PER_DAY

UNCERTAINTY_SCENARIOS_PATH = ROOT / "inputs" / "uncertainty_scenarios.json"
RESCALE_TARGET_MAX_PCT = 50.0


SCENARIO_OVERRIDES_ENV = "PLANNER_SCENARIO_OVERRIDES"


@lru_cache(maxsize=1)
def load_uncertainty_scenarios() -> dict:
    scenarios = json.loads(UNCERTAINTY_SCENARIOS_PATH.read_text())["scenarios"]
    override_path = os.environ.get(SCENARIO_OVERRIDES_ENV)
    if override_path:
        for name, edit in json.loads(Path(override_path).read_text())["scenarios"].items():
            if name in scenarios:
                scenarios[name].update(edit)
    return scenarios


@lru_cache(maxsize=1)
def _global_max_error_pct() -> float:
    """Largest wind/solar capacity_scale error (%) across every unc scenario/country, used
    as the rescale reference point so RESCALE_TARGET_MAX_PCT corresponds to that worst case."""
    scenarios = load_uncertainty_scenarios()
    max_err = 0.0
    for name, sc in scenarios.items():
        if name == "p100":
            continue
        for resource in ("wind", "solar"):
            for scale in sc[resource].values():
                err = (1.0 - scale) * 100.0
                if err > max_err:
                    max_err = err
    return max_err


def _rescaled_capacity_scale(scenario: str | None, country: str) -> tuple[float, float]:
    """(wind_scale, solar_scale) for this scenario/country, rescaled so the worst case across
    every unc scenario/country corresponds to RESCALE_TARGET_MAX_PCT error (not the real,
    larger error baked into the training data) -- (1.0, 1.0) if scenario is None/unknown or
    has no entry for this country (e.g. 'p100'). Only used for the candidate's OWN available
    capacity (and hence its own contribution to the price-model input); the rest of the
    system's price features keep reflecting the real, un-rescaled scenario severity, since
    those come straight from edf/hdf's real per-scenario dispatch data."""
    if not scenario:
        return 1.0, 1.0
    sc = load_uncertainty_scenarios().get(scenario)
    if sc is None:
        return 1.0, 1.0
    factor = RESCALE_TARGET_MAX_PCT / _global_max_error_pct()

    def rescale(raw_scale: float) -> float:
        err = (1.0 - raw_scale) * 100.0
        return 1.0 - (err * factor) / 100.0

    wind_scale = rescale(sc["wind"].get(country, 1.0))
    solar_scale = rescale(sc["solar"].get(country, 1.0))
    return wind_scale, solar_scale


def _month_boundaries(total_days: int = TOTAL_YEAR_DAYS, n_months: int = N_MONTHS) -> list[int]:
    """``n_months + 1`` cut points partitioning ``[0, total_days)`` into ``n_months`` chunks."""
    return [round(i * total_days / n_months) for i in range(n_months + 1)]


def representative_days(n: int, total_days: int = TOTAL_YEAR_DAYS, n_months: int = N_MONTHS,
                        ) -> tuple[list[int], list[float]]:
    """``n`` evenly-spaced representative days per month-chunk, plus each day's weight; returns ``(days, weights)``."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    bounds = _month_boundaries(total_days, n_months)
    days: list[int] = []
    weights: list[float] = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        length = hi - lo
        if n >= length:
            days.extend(range(lo + 1, hi + 1))
            weights.extend([1.0] * length)
            continue
        picks = np.unique(np.round(np.linspace(lo, hi - 1, n)).astype(int))
        w = length / len(picks)
        days.extend(int(p) + 1 for p in picks)
        weights.extend([w] * len(picks))
    return days, weights


def first_n_days(n: int, total_days: int = TOTAL_YEAR_DAYS, n_months: int = N_MONTHS,
                 ) -> tuple[list[int], list[float]]:
    """Same weighting scheme as ``representative_days``, but picks the first ``n`` calendar days of each month-chunk."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    bounds = _month_boundaries(total_days, n_months)
    days: list[int] = []
    weights: list[float] = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        length = hi - lo
        if n >= length:
            days.extend(range(lo + 1, hi + 1))
            weights.extend([1.0] * length)
            continue
        w = length / n
        days.extend(range(lo + 1, lo + 1 + n))
        weights.extend([w] * n)
    return days, weights

_ELEC_ZONE_OVERRIDES = {"BE": "BE00", "NL": "NL00"}

_CAPACITY_OVERRIDE_FIELDS = {
    "electrolyser_mw": "g_investor_electrolyser_mw_overrides",
    "wind_mw": "g_investor_wind_mw_overrides",
    "pv_mw": "g_investor_pv_mw_overrides",
    "battery_mw": "g_investor_battery_mw_overrides",
    "tank_mw": "g_investor_tank_mw_overrides",
    # Optional: the actual MWh of the battery/tank candidate mix chosen by the capacity
    # planner, overriding the fixed mw * duration_hours default (see _g_investor_sizing).
    "battery_mwh": "g_investor_battery_mwh_overrides",
    "tank_mwh": "g_investor_tank_mwh_overrides",
}

TANK_EFFICIENCY = 0.99


def _run_config(zone: str, start_day: int, end_day: int, capacities: dict | None = None) -> RunConfig:
    overrides = {}
    if capacities:
        country = zone[:2]
        for key, field in _CAPACITY_OVERRIDE_FIELDS.items():
            if key in capacities:
                overrides[field] = {country: float(capacities[key])}
    return RunConfig(
        zones=[zone], start_day=start_day, end_day=end_day,
        zones_db=ROOT / "inputs" / "zones_2030.parquet",
        networks_db=ROOT / "inputs" / "networks_2030.parquet",
        g_investor_tank_efficiency=TANK_EFFICIENCY,
        **overrides,
    )


def _donor_candidates(country: str, host_zone: str, resource_idx: int,
                      profile_info: dict, all_zones: list[str]) -> list[str]:
    """Zone(s) that could supply this resource's (wind=0/solar=1) weather profile for ``host_zone``."""
    def has_data(z):
        return profile_info.get(z, (False, 0.0, False, 0.0))[resource_idx * 2]
    if has_data(host_zone):
        return [host_zone]
    sibs = [z for z in all_zones if z[:2] == country and z != host_zone and has_data(z)]
    return sibs or [host_zone]


def _donor_zone(candidates: list[str], cap_key: str, zdata: dict) -> str:
    """Best-capacity zone among ``candidates`` (see ``_donor_candidates``)."""
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda z: zdata[z].capacities.get(cap_key, 0.0))


@lru_cache(maxsize=64)
def _zone_raw_profile(zone: str, hours_key: tuple[int, ...]):
    """Capacity- and scenario-independent part of ``sizing_and_profiles``: host-zone validation,
    wind/PV donor resolution, and the raw (un-rescaled) normalized capacity-factor profiles.
    Cached because it's invariant across every scenario/trial-point/core-point call within a
    Benders run for a given zone and representative-hour selection -- only the scenario rescale
    and the candidate's own MW sizing (applied by the caller) actually vary call to call."""
    hours = np.array(hours_key, dtype=int)
    start_day = int(hours.min()) // HOURS_PER_DAY + 1
    end_day = int(hours.max()) // HOURS_PER_DAY + 1
    cfg = _run_config(zone, start_day, end_day, None)
    country = zone[:2]
    host_zone = ed_model._h2_main_zones(cfg)[country]
    if host_zone != zone:
        raise ValueError(f"{zone} is not {country}'s main H2 zone (that's {host_zone}); "
                         f"pass the main H2 zone instead.")

    profile_info = ed_model._g_investor_renewable_profile_info(str(cfg.zones_db))
    all_zones = discover_zones(cfg.zones_db)

    wind_candidates = _donor_candidates(country, host_zone, 0, profile_info, all_zones)
    pv_candidates = _donor_candidates(country, host_zone, 1, profile_info, all_zones)
    needed_zones = sorted({host_zone, *wind_candidates, *pv_candidates})

    zdata = ed_dl.load_zones_from_db(needed_zones, cfg.zones_db, 0, TOTAL_YEAR_HOURS)
    wind_donor = _donor_zone(wind_candidates, "Wind (onshore) (MW)", zdata)
    pv_donor = _donor_zone(pv_candidates, "Solar (MW)", zdata)

    wind_max = max(profile_info.get(wind_donor, (False, 0.0, False, 0.0))[1], 1e-9)
    pv_max = max(profile_info.get(pv_donor, (False, 0.0, False, 0.0))[3], 1e-9)
    wind_cf_full = zdata[wind_donor].profiles["Wind_Onshore Profile"].to_numpy(dtype=float)
    pv_cf_full = zdata[pv_donor].profiles["Solar Profile"].to_numpy(dtype=float)
    wind_cf = np.clip(wind_cf_full[hours], 0.0, None)
    pv_cf = np.clip(pv_cf_full[hours], 0.0, None)
    wind_cf_norm = np.clip(wind_cf / wind_max, 0.0, 1.0)
    pv_cf_norm = np.clip(pv_cf / pv_max, 0.0, 1.0)
    return host_zone, wind_donor, pv_donor, wind_cf_norm, pv_cf_norm


def sizing_and_profiles(zone: str, hours: np.ndarray, capacities: dict | None = None,
                        scenario: str | None = None):
    """Resolve this country's Investor sizing plus wind/PV availability upper bounds for the
    given year-hour positions. If ``scenario`` is given, the candidate's OWN wind/PV capacity
    factor (and hence its own available-capacity bound and its own contribution to the price
    feature row) is derated by that scenario's country-level capacity_scale, rescaled so the
    worst case across all scenarios/countries corresponds to RESCALE_TARGET_MAX_PCT error --
    NOT the real, larger error baked into the training data (see ``_rescaled_capacity_scale``).
    The rest of the system's price features are untouched, still reflecting the real severity."""
    start_day = int(hours.min()) // HOURS_PER_DAY + 1
    end_day = int(hours.max()) // HOURS_PER_DAY + 1
    cfg = _run_config(zone, start_day, end_day, capacities)
    country = zone[:2]
    sizing = ed_model._g_investor_sizing(cfg)[country]

    hours_key = tuple(int(h) for h in hours)
    host_zone, wind_donor, pv_donor, wind_cf_norm, pv_cf_norm = _zone_raw_profile(zone, hours_key)

    wind_scale, pv_scale = _rescaled_capacity_scale(scenario, country)
    wind_cf_norm = wind_cf_norm * wind_scale
    pv_cf_norm = pv_cf_norm * pv_scale

    wind_upper = wind_cf_norm * sizing["wind_mw"]
    pv_upper = pv_cf_norm * sizing["pv_mw"]
    return cfg, sizing, host_zone, wind_donor, pv_donor, wind_upper, pv_upper, wind_cf_norm, pv_cf_norm


_BACKTEST_SCENARIO = "p100"


def _backfill_required_price_cols(df: pd.DataFrame, raw_df: pd.DataFrame, target_col: str,
                                  commodity: str) -> pd.DataFrame:
    """Ensure every ``price_<zone>`` column any trained zone model actually needs is present.

    ``add_candidate_neighbor_prices`` recomputes each zone's top-5 correlated neighbours from
    whatever (possibly one-scenario, possibly unusual) data it's given -- a scenario extreme
    enough to shift that correlation ranking can silently drop a column the trained model still
    expects (KeyError at predict time). The model's own required feature list is fixed at
    training time, so we just backfill any of ITS needed ``price_<zone>`` columns straight from
    ``raw_df`` (same scenario, so ``hour`` keys are unique) regardless of this scenario's own
    correlation structure.
    """
    bundle = price_api._bundle(commodity)
    needed_zones = {n[len("price_"):] for z in bundle["zones"] for n in bundle["zones"][z]["features"]
                    if n.startswith("price_")}
    missing = [z for z in needed_zones if f"price_{z}" not in df.columns and z in raw_df["zone"].values]
    if not missing:
        return df
    wide = raw_df[raw_df["zone"].isin(missing)].pivot_table(index="hour", columns="zone", values=target_col,
                                                            aggfunc="first")
    for z in missing:
        df[f"price_{z}"] = df["hour"].map(wide[z])
    return df


def enriched_elec_df(scenario: str = _BACKTEST_SCENARIO) -> pd.DataFrame:
    """Full elec_samples.parquet enriched with neighbour/candidate-price columns, filtered to one capacity scenario."""
    ROOT_IN = ROOT / "inputs"
    ROOT_EXCHANGE = ROOT / "data_exchange" / "01_dispatch_output__train_input"
    edf = pd.read_parquet(ROOT_EXCHANGE / "elec_samples.parquet")
    if "scenario" in edf.columns:
        edf = edf[edf["scenario"] == scenario].drop(columns="scenario")
    raw_edf = edf
    eadj = load_adjacency(ROOT_IN / "elec_adjacency.json")
    edf, _ = add_neighbor_features(edf, "demand", eadj, "residual_load")
    edf, _ = add_candidate_neighbor_prices(edf, "price_eur_mwh", eadj)
    edf = _backfill_required_price_cols(edf, raw_edf, "price_eur_mwh", "electricity")
    return edf


def enriched_h2_df(scenario: str = _BACKTEST_SCENARIO) -> pd.DataFrame:
    """Full h2_samples.parquet enriched with neighbour/candidate-price columns, filtered to one capacity scenario."""
    ROOT_IN = ROOT / "inputs"
    ROOT_EXCHANGE = ROOT / "data_exchange" / "01_dispatch_output__train_input"
    hdf = pd.read_parquet(ROOT_EXCHANGE / "h2_samples.parquet")
    if "scenario" in hdf.columns:
        hdf = hdf[hdf["scenario"] == scenario].drop(columns="scenario")
    raw_hdf = hdf
    hadj = load_adjacency(ROOT_IN / "h2_adjacency.json")
    hdf, _ = add_neighbor_features(hdf, "h2_demand", hadj, None)
    hdf, _ = add_candidate_neighbor_prices(hdf, "h2_price", hadj)
    hdf = _backfill_required_price_cols(hdf, raw_hdf, "h2_price", "hydrogen")
    return hdf


def _capacity_scale(values: np.ndarray, capacity_mw: np.ndarray, add_mw: float) -> np.ndarray:
    """Scale a real activity series by (scenario capacity + candidate add) / scenario capacity; no-op where capacity is 0."""
    cap = np.asarray(capacity_mw, dtype=float)
    factor = np.divide(cap + add_mw, cap, out=np.ones_like(cap), where=cap > 0)
    return values * factor


def proxy_price_series(elec_zone: str, hours: np.ndarray, h2_zone: str | None = None,
                       edf: pd.DataFrame | None = None, hdf: pd.DataFrame | None = None,
                       wind_gen_add: np.ndarray | None = None, pv_gen_add: np.ndarray | None = None,
                       battery_add: float = 0.0, electrolyser_add: float = 0.0,
                       tank_add: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Predict (elec_price, h2_price) for the given year-hour positions from the trained proxy models, optionally reflecting the candidate's own added wind/PV/battery/electrolyser/tank capacity in the feature row first."""
    if h2_zone is None:
        h2_zone = elec_zone
    hours = list(hours)
    H = len(hours)

    elec_bundle = price_api._bundle("electricity")
    h2_bundle = price_api._bundle("hydrogen")

    if edf is None:
        edf = enriched_elec_df()
    erow = edf[(edf["zone"] == elec_zone) & (edf["hour"].isin(hours))].sort_values("hour")
    efeats = elec_bundle["zones"][elec_zone]["features"]
    has_add = (wind_gen_add is not None or pv_gen_add is not None
              or battery_add or electrolyser_add or tank_add)
    if has_add:
        assert (wind_gen_add is None or len(wind_gen_add) == H) and \
              (pv_gen_add is None or len(pv_gen_add) == H), \
              f"wind_gen_add/pv_gen_add must have length {H}, matching hours"
        w_add = np.zeros(H) if wind_gen_add is None else np.asarray(wind_gen_add, dtype=float)
        s_add = np.zeros(H) if pv_gen_add is None else np.asarray(pv_gen_add, dtype=float)
        erow = erow.copy()
        if "wind" in erow.columns:
            erow["wind"] = erow["wind"].to_numpy() + w_add
        if "solar" in erow.columns:
            erow["solar"] = erow["solar"].to_numpy() + s_add
        if "residual_load" in erow.columns:
            erow["residual_load"] = erow["residual_load"].to_numpy() - w_add - s_add
        if "wind_capacity_mw" in erow.columns:
            erow["wind_capacity_mw"] = erow["wind_capacity_mw"].to_numpy() + w_add.max(initial=0.0)
        if "pv_capacity_mw" in erow.columns:
            erow["pv_capacity_mw"] = erow["pv_capacity_mw"].to_numpy() + s_add.max(initial=0.0)
        if battery_add and "battery" in erow.columns and "battery_capacity_mw" in erow.columns:
            erow["battery"] = _capacity_scale(erow["battery"].to_numpy(),
                                              erow["battery_capacity_mw"].to_numpy(), battery_add)
    p_elec = model_predict(elec_bundle, elec_zone, erow[efeats])

    if hdf is None:
        hdf = enriched_h2_df()
    hrow = hdf[(hdf["zone"] == h2_zone) & (hdf["hour"].isin(hours))].sort_values("hour")
    hfeats = h2_bundle["zones"][h2_zone]["features"]
    if has_add:
        hrow = hrow.copy()
        if "wind_capacity_mw" in hrow.columns:
            hrow["wind_capacity_mw"] = hrow["wind_capacity_mw"].to_numpy() + w_add.max(initial=0.0)
        if "pv_capacity_mw" in hrow.columns:
            hrow["pv_capacity_mw"] = hrow["pv_capacity_mw"].to_numpy() + s_add.max(initial=0.0)
        if electrolyser_add and "electrolyser_gen" in hrow.columns and "electrolyser_capacity_mw" in hrow.columns:
            hrow["electrolyser_gen"] = _capacity_scale(hrow["electrolyser_gen"].to_numpy(),
                                                        hrow["electrolyser_capacity_mw"].to_numpy(), electrolyser_add)
        if tank_add and "storage" in hrow.columns and "tank_capacity_mw" in hrow.columns:
            hrow["storage"] = _capacity_scale(hrow["storage"].to_numpy(),
                                              hrow["tank_capacity_mw"].to_numpy(), tank_add)
    p_h2 = model_predict(h2_bundle, h2_zone, hrow[hfeats])

    assert len(p_elec) == H and len(p_h2) == H, f"expected {H} hours, got {len(p_elec)}/{len(p_h2)}"
    return np.asarray(p_elec, dtype=float), np.asarray(p_h2, dtype=float)


def _capacity_bounded(m: linopy.Model, name: str, coords: list[pd.Index], upper_val,
                      return_duals: bool, cap_name: str):
    """Build a [0, ``upper_val``]-bounded variable, using a named constraint instead of a plain bound when duals are needed."""
    if not return_duals:
        return m.add_variables(lower=0.0, upper=upper_val, coords=coords, name=name)
    v = m.add_variables(lower=0.0, coords=coords, name=name)
    m.add_constraints(v <= upper_val, name=cap_name)
    return v


def solve(zone: str, start_day: int | None = None, end_day: int | None = None,
         rep_days_per_month: int | None = None,
         capacities: dict | None = None, return_duals: bool = False,
         edf: pd.DataFrame | None = None, hdf: pd.DataFrame | None = None,
         day_selection: str = "first", quiet: bool = False,
         scenario: str | None = None) -> pd.DataFrame:
    """Solve one zone's General Investor LP over a contiguous day range or representative-day sample, priced by the proxy models.
    ``scenario``, if given, derates the candidate's OWN wind/PV capacity (rescaled, see
    ``sizing_and_profiles``) -- pass the same scenario name used to build ``edf``/``hdf``."""
    representative = rep_days_per_month is not None
    if representative:
        if start_day is not None or end_day is not None:
            raise ValueError("rep_days_per_month is mutually exclusive with "
                             "start_day/end_day -- representative sampling always "
                             "spans the full 364-day year.")
        if day_selection == "first":
            days, day_weights = first_n_days(rep_days_per_month)
        else:
            days, day_weights = representative_days(rep_days_per_month)
    else:
        if start_day is None:
            raise ValueError("either start_day (contiguous mode) or "
                             "rep_days_per_month (representative-sampling mode) is required.")
        if end_day is None:
            end_day = start_day
        days = list(range(start_day, end_day + 1))
        day_weights = [1.0] * len(days)

    n_days = len(days)
    hours = np.array([(d - 1) * HOURS_PER_DAY + h for d in days for h in range(HOURS_PER_DAY)], dtype=int)
    H = len(hours)

    t0 = time.time()
    (cfg, s, host_zone, wind_donor, pv_donor, wind_upper, pv_upper,
    wind_cf_norm, pv_cf_norm) = sizing_and_profiles(zone, hours, capacities, scenario)
    elec_zone = _ELEC_ZONE_OVERRIDES.get(zone[:2], zone)
    p_elec, p_h2 = proxy_price_series(elec_zone, hours, h2_zone=zone, edf=edf, hdf=hdf,
                                      wind_gen_add=wind_cf_norm * s["wind_mw"],
                                      pv_gen_add=pv_cf_norm * s["pv_mw"],
                                      battery_add=s["battery_mw"], electrolyser_add=s["electrolyser_mw"],
                                      tank_add=s["tank_mw"])

    ely_mw, batt_mw, batt_mwh = s["electrolyser_mw"], s["battery_mw"], s["battery_mwh"]
    tank_mw, tank_mwh = s["tank_mw"], s["tank_mwh"]
    ely_eff = cfg.g_investor_electrolyser_efficiency
    batt_eff, tank_eff = cfg.g_investor_battery_efficiency, cfg.g_investor_tank_efficiency
    sto_cost = cfg.storage_op_cost_eur_per_mwh
    batt_soc0 = cfg.initial_soc_fraction * batt_mwh
    tank_soc0 = cfg.initial_soc_fraction * tank_mwh
    annual_hours = TOTAL_YEAR_HOURS if representative else H

    m = linopy.Model()

    if not representative:
        hour_idx = pd.RangeIndex(H, name="hour")
        coords = [hour_idx]
        wind_p = _capacity_bounded(m, "wind_p", coords, wind_upper, return_duals, "wind_cap")
        pv_p = _capacity_bounded(m, "pv_p", coords, pv_upper, return_duals, "pv_cap")
        batt_dis = _capacity_bounded(m, "batt_dis", coords, batt_mw, return_duals, "batt_dis_cap")
        batt_ch = _capacity_bounded(m, "batt_ch", coords, batt_mw, return_duals, "batt_ch_cap")
        tank_dis = _capacity_bounded(m, "tank_dis", coords, tank_mw, return_duals, "tank_dis_cap")
        tank_ch = _capacity_bounded(m, "tank_ch", coords, tank_mw, return_duals, "tank_ch_cap")
        batt_soc = _capacity_bounded(m, "batt_soc", coords, batt_mwh, return_duals, "batt_soc_cap")
        tank_soc = _capacity_bounded(m, "tank_soc", coords, tank_mwh, return_duals, "tank_soc_cap")
        ely_p = _capacity_bounded(m, "ely_p", coords, ely_mw, return_duals, "ely_cap")

        m.add_constraints(batt_soc.isel(hour=0) - batt_soc0 - batt_eff * batt_ch.isel(hour=0) + batt_dis.isel(hour=0) == 0,
                          name="batt_balance_0")
        m.add_constraints(batt_soc.isel(hour=slice(1, None)) - batt_soc.isel(hour=slice(None, -1))
                          - batt_eff * batt_ch.isel(hour=slice(1, None)) + batt_dis.isel(hour=slice(1, None)) == 0,
                          name="batt_balance")
        m.add_constraints(tank_soc.isel(hour=0) - tank_soc0 - tank_eff * tank_ch.isel(hour=0) + tank_dis.isel(hour=0) == 0,
                          name="tank_balance_0")
        m.add_constraints(tank_soc.isel(hour=slice(1, None)) - tank_soc.isel(hour=slice(None, -1))
                          - tank_eff * tank_ch.isel(hour=slice(1, None)) + tank_dis.isel(hour=slice(1, None)) == 0,
                          name="tank_balance")
        if cfg.cyclic_storage:
            m.add_constraints(batt_soc.isel(hour=-1) >= batt_soc0, name="batt_cyclic")
            m.add_constraints(tank_soc.isel(hour=-1) >= tank_soc0, name="tank_cyclic")

        cost = (-(p_elec * (wind_p + pv_p + batt_dis - batt_ch - ely_p)).sum()
               - (p_h2 * (ely_eff * ely_p + tank_dis - tank_ch)).sum()
               + sto_cost * (batt_ch.sum() + batt_dis.sum() + tank_ch.sum() + tank_dis.sum()))
        m.add_objective(cost)
    else:
        day_idx = pd.RangeIndex(n_days, name="day")
        hid_idx = pd.RangeIndex(HOURS_PER_DAY, name="hid")
        coords = [day_idx, hid_idx]

        def r2d(a):
            return np.asarray(a).reshape(n_days, HOURS_PER_DAY)

        wind_upper2d, pv_upper2d = r2d(wind_upper), r2d(pv_upper)
        p_elec2d, p_h2_2d = r2d(p_elec), r2d(p_h2)
        w = xr.DataArray(np.repeat(np.asarray(day_weights, dtype=float), HOURS_PER_DAY).reshape(n_days, HOURS_PER_DAY),
                         coords=coords)

        wind_p = _capacity_bounded(m, "wind_p", coords, wind_upper2d, return_duals, "wind_cap")
        pv_p = _capacity_bounded(m, "pv_p", coords, pv_upper2d, return_duals, "pv_cap")
        batt_dis = _capacity_bounded(m, "batt_dis", coords, batt_mw, return_duals, "batt_dis_cap")
        batt_ch = _capacity_bounded(m, "batt_ch", coords, batt_mw, return_duals, "batt_ch_cap")
        tank_dis = _capacity_bounded(m, "tank_dis", coords, tank_mw, return_duals, "tank_dis_cap")
        tank_ch = _capacity_bounded(m, "tank_ch", coords, tank_mw, return_duals, "tank_ch_cap")
        batt_soc = _capacity_bounded(m, "batt_soc", coords, batt_mwh, return_duals, "batt_soc_cap")
        tank_soc = _capacity_bounded(m, "tank_soc", coords, tank_mwh, return_duals, "tank_soc_cap")
        ely_p = _capacity_bounded(m, "ely_p", coords, ely_mw, return_duals, "ely_cap")

        m.add_constraints(batt_soc.isel(hid=0) - batt_soc0 - batt_eff * batt_ch.isel(hid=0) + batt_dis.isel(hid=0) == 0,
                          name="batt_balance_0")
        m.add_constraints(batt_soc.isel(hid=slice(1, None)) - batt_soc.isel(hid=slice(None, -1))
                          - batt_eff * batt_ch.isel(hid=slice(1, None)) + batt_dis.isel(hid=slice(1, None)) == 0,
                          name="batt_balance")
        m.add_constraints(tank_soc.isel(hid=0) - tank_soc0 - tank_eff * tank_ch.isel(hid=0) + tank_dis.isel(hid=0) == 0,
                          name="tank_balance_0")
        m.add_constraints(tank_soc.isel(hid=slice(1, None)) - tank_soc.isel(hid=slice(None, -1))
                          - tank_eff * tank_ch.isel(hid=slice(1, None)) + tank_dis.isel(hid=slice(1, None)) == 0,
                          name="tank_balance")
        if cfg.cyclic_storage:
            m.add_constraints(batt_soc.isel(hid=-1) >= batt_soc0, name="batt_cyclic")
            m.add_constraints(tank_soc.isel(hid=-1) >= tank_soc0, name="tank_cyclic")

        cost = (-(w * p_elec2d * (wind_p + pv_p + batt_dis - batt_ch - ely_p)).sum()
               - (w * p_h2_2d * (ely_eff * ely_p + tank_dis - tank_ch)).sum()
               + sto_cost * ((w * batt_ch).sum() + (w * batt_dis).sum()
                            + (w * tank_ch).sum() + (w * tank_dis).sum()))
        m.add_objective(cost)

    build_s = time.time() - t0
    t1 = time.time()
    solver_kwargs = {"output_flag": False} if quiet else {}
    status, condition = m.solve(solver_name="highs", **solver_kwargs)
    solve_s = time.time() - t1
    if status != "ok":
        raise RuntimeError(f"solve failed: {status}/{condition}")

    cut_coeffs = None
    if return_duals:
        mu_electrolyser = float(m.dual["ely_cap"].sum())
        wind_cf_shaped = wind_cf_norm.reshape(n_days, HOURS_PER_DAY) if representative else wind_cf_norm
        pv_cf_shaped = pv_cf_norm.reshape(n_days, HOURS_PER_DAY) if representative else pv_cf_norm
        mu_wind = float((m.dual["wind_cap"] * wind_cf_shaped).sum())
        mu_pv = float((m.dual["pv_cap"] * pv_cf_shaped).sum())
        batt_soc0_terms = float(m.dual["batt_balance_0"].sum())
        tank_soc0_terms = float(m.dual["tank_balance_0"].sum())
        if cfg.cyclic_storage:
            batt_soc0_terms += float(m.dual["batt_cyclic"].sum())
            tank_soc0_terms += float(m.dual["tank_cyclic"].sum())
        mu_battery = (float(m.dual["batt_dis_cap"].sum()) + float(m.dual["batt_ch_cap"].sum())
                     + float(m.dual["batt_soc_cap"].sum()) * cfg.g_investor_battery_duration_hours
                     + batt_soc0_terms * cfg.initial_soc_fraction * cfg.g_investor_battery_duration_hours)
        mu_tank = (float(m.dual["tank_dis_cap"].sum()) + float(m.dual["tank_ch_cap"].sum())
                  + float(m.dual["tank_soc_cap"].sum()) * cfg.g_investor_tank_duration_hours
                  + tank_soc0_terms * cfg.initial_soc_fraction * cfg.g_investor_tank_duration_hours)
        cut_coeffs = {"electrolyser_mw": mu_electrolyser, "wind_mw": mu_wind, "pv_mw": mu_pv,
                     "battery_mw": mu_battery, "tank_mw": mu_tank}

    sol = m.solution
    wind_v = np.asarray(sol["wind_p"].values).reshape(-1)
    pv_v = np.asarray(sol["pv_p"].values).reshape(-1)
    batt_dis_v = np.asarray(sol["batt_dis"].values).reshape(-1)
    batt_ch_v = np.asarray(sol["batt_ch"].values).reshape(-1)
    ely_p_v = np.asarray(sol["ely_p"].values).reshape(-1)
    tank_dis_v = np.asarray(sol["tank_dis"].values).reshape(-1)
    tank_ch_v = np.asarray(sol["tank_ch"].values).reshape(-1)
    grid_exchange_v = wind_v + pv_v + batt_dis_v - batt_ch_v - ely_p_v
    pipeline_exchange_v = ely_eff * ely_p_v + tank_dis_v - tank_ch_v
    out = pd.DataFrame({
        "hour": np.arange(H),
        "H2 Producer wind (MW)": wind_v,
        "H2 Producer pv (MW)": pv_v,
        "H2 Producer battery discharge (MW)": batt_dis_v,
        "H2 Producer battery charge (-) (MW)": -batt_ch_v,
        "H2 Producer electrolyser load (-) (MW)": -ely_p_v,
        "H2 Producer grid exchange (MW)": grid_exchange_v,
        "H2 Producer electrolyser production (MW)": ely_eff * ely_p_v,
        "H2 Producer tank discharge (MW)": tank_dis_v,
        "H2 Producer tank charge (-) (MW)": -tank_ch_v,
        "H2 Producer pipeline exchange (MW)": pipeline_exchange_v,
    })
    if representative:
        out["day_of_year"] = np.repeat(days, HOURS_PER_DAY)
        out["day_weight"] = np.repeat(day_weights, HOURS_PER_DAY)
    out.attrs["objective"] = float(m.objective.value)
    out.attrs["p_elec"] = p_elec
    out.attrs["p_h2"] = p_h2
    out.attrs["host_zone"] = host_zone
    out.attrs["sizing"] = s
    out.attrs["build_seconds"] = build_s
    out.attrs["solve_seconds"] = solve_s
    out.attrs["n_hours"] = H
    out.attrs["annualized_hours"] = annual_hours
    out.attrs["rep_days_per_month"] = rep_days_per_month
    out.attrs["sampled_days"] = days if representative else None
    out.attrs["capacities"] = {"electrolyser_mw": ely_mw, "wind_mw": s["wind_mw"], "pv_mw": s["pv_mw"],
                              "battery_mw": batt_mw, "tank_mw": tank_mw}
    out.attrs["cut_coeffs"] = cut_coeffs
    return out


# Capacity-bounded variable name -> (cap constraint name, upper-bound DataArray key). Shared
# between the first build and every later in-place update so the two stay in lockstep.
_CAP_VARS = [
    ("wind_p", "wind_cap", "wind_upper_da"),
    ("pv_p", "pv_cap", "pv_upper_da"),
    ("batt_dis", "batt_dis_cap", "batt_mw_da"),
    ("batt_ch", "batt_ch_cap", "batt_mw_da"),
    ("tank_dis", "tank_dis_cap", "tank_mw_da"),
    ("tank_ch", "tank_ch_cap", "tank_mw_da"),
    ("batt_soc", "batt_soc_cap", "batt_mwh_da"),
    ("tank_soc", "tank_soc_cap", "tank_mwh_da"),
    ("ely_p", "ely_cap", "ely_mw_da"),
]

# Persistent (zones, rep_days_per_month, day_selection, scenario, return_duals) -> built model/solver.
# Reused across Benders iterations: rebuilding the linopy Model (add_variables/add_constraints/
# export-to-solver) dominates solve_joint's "build" time (see profiling in the Oct 2026 Benders
# speed-up), while only capacity-dependent bounds, balance-constraint RHS, and price-weighted
# objective coefficients actually change iteration to iteration. We mutate those in place instead.
_JOINT_SUBPROBLEM_CACHE: dict[tuple, dict] = {}


def reset_joint_model_cache() -> None:
    """Drop every cached persistent Benders subproblem model (closes its HiGHS instance first).
    Call between unrelated runs sharing a process, or to force a clean rebuild for benchmarking."""
    for state in _JOINT_SUBPROBLEM_CACHE.values():
        try:
            state["solver"].close()
        except Exception:
            pass
    _JOINT_SUBPROBLEM_CACHE.clear()


def solve_joint(zones: list[str], capacities: dict[str, dict], rep_days_per_month: int,
                return_duals: bool = True,
                edf: pd.DataFrame | None = None, hdf: pd.DataFrame | None = None,
                day_selection: str = "first", quiet: bool = False,
                scenario: str | None = None, reuse: bool = True) -> dict:
    """Solve every zone's General Investor LP jointly (representative-day horizon only) for a
    given trial capacity vector. ``scenario``, if given, derates every zone's own wind/PV
    capacity (rescaled, see ``sizing_and_profiles``) -- pass the same scenario name used to
    build ``edf``/``hdf``. If ``reuse`` (default), the LP for this exact
    (zones, rep_days_per_month, day_selection, scenario, return_duals) combination is built once
    and kept warm across calls (e.g. Benders iterations): later calls only mutate capacity-driven
    bounds/RHS/objective coefficients in place and re-solve via HiGHS's own warm-started direct
    API, instead of rebuilding and re-exporting the whole LP from scratch every time."""
    t0 = time.time()
    if day_selection == "first":
        days, day_weights = first_n_days(rep_days_per_month)
    else:
        days, day_weights = representative_days(rep_days_per_month)
    n_days = len(days)
    hours = np.array([(d - 1) * HOURS_PER_DAY + h for d in days for h in range(HOURS_PER_DAY)], dtype=int)
    H = len(hours)
    annual_hours = TOTAL_YEAR_HOURS

    zone_idx = pd.Index(zones, name="zone")
    day_idx = pd.RangeIndex(n_days, name="day")
    hid_idx = pd.RangeIndex(HOURS_PER_DAY, name="hid")
    coords = [zone_idx, day_idx, hid_idx]

    w = xr.DataArray(np.repeat(np.asarray(day_weights, dtype=float), HOURS_PER_DAY).reshape(n_days, HOURS_PER_DAY),
                     coords=[day_idx, hid_idx])

    cfgs, ely_effs, ely_mws, batt_mws, batt_mwhs, tank_mws, tank_mwhs = {}, {}, {}, {}, {}, {}, {}
    wind_uppers, pv_uppers, wind_cf_norms, pv_cf_norms, p_elecs, p_h2s = [], [], [], [], [], []
    host_zones = {}
    for z in zones:
        (cfg, s, host_zone, wind_donor, pv_donor, wind_upper, pv_upper,
        wind_cf_norm, pv_cf_norm) = sizing_and_profiles(z, hours, capacities[z], scenario)
        cfgs[z] = cfg
        host_zones[z] = host_zone
        ely_effs[z] = cfg.g_investor_electrolyser_efficiency
        ely_mws[z] = s["electrolyser_mw"]
        batt_mws[z], batt_mwhs[z] = s["battery_mw"], s["battery_mwh"]
        tank_mws[z], tank_mwhs[z] = s["tank_mw"], s["tank_mwh"]
        wind_uppers.append(wind_upper.reshape(n_days, HOURS_PER_DAY))
        pv_uppers.append(pv_upper.reshape(n_days, HOURS_PER_DAY))
        wind_cf_norms.append(wind_cf_norm.reshape(n_days, HOURS_PER_DAY))
        pv_cf_norms.append(pv_cf_norm.reshape(n_days, HOURS_PER_DAY))
        elec_zone = _ELEC_ZONE_OVERRIDES.get(z[:2], z)
        p_elec, p_h2 = proxy_price_series(elec_zone, hours, h2_zone=z, edf=edf, hdf=hdf,
                                          wind_gen_add=wind_cf_norm * s["wind_mw"],
                                          pv_gen_add=pv_cf_norm * s["pv_mw"],
                                          battery_add=s["battery_mw"], electrolyser_add=s["electrolyser_mw"],
                                          tank_add=s["tank_mw"])
        p_elecs.append(p_elec.reshape(n_days, HOURS_PER_DAY))
        p_h2s.append(p_h2.reshape(n_days, HOURS_PER_DAY))

    wind_upper_da = xr.DataArray(np.stack(wind_uppers), coords=coords)
    pv_upper_da = xr.DataArray(np.stack(pv_uppers), coords=coords)
    wind_cf_norm_da = xr.DataArray(np.stack(wind_cf_norms), coords=coords)
    pv_cf_norm_da = xr.DataArray(np.stack(pv_cf_norms), coords=coords)
    p_elec_da = xr.DataArray(np.stack(p_elecs), coords=coords)
    p_h2_da = xr.DataArray(np.stack(p_h2s), coords=coords)

    ely_mw_da = xr.DataArray([ely_mws[z] for z in zones], coords=[zone_idx])
    ely_eff_da = xr.DataArray([ely_effs[z] for z in zones], coords=[zone_idx])
    batt_mw_da = xr.DataArray([batt_mws[z] for z in zones], coords=[zone_idx])
    batt_mwh_da = xr.DataArray([batt_mwhs[z] for z in zones], coords=[zone_idx])
    tank_mw_da = xr.DataArray([tank_mws[z] for z in zones], coords=[zone_idx])
    tank_mwh_da = xr.DataArray([tank_mwhs[z] for z in zones], coords=[zone_idx])

    cfg0 = cfgs[zones[0]]
    batt_eff, tank_eff = cfg0.g_investor_battery_efficiency, cfg0.g_investor_tank_efficiency
    sto_cost = cfg0.storage_op_cost_eur_per_mwh
    batt_soc0_da = cfg0.initial_soc_fraction * batt_mwh_da
    tank_soc0_da = cfg0.initial_soc_fraction * tank_mwh_da
    upper_das = {"wind_upper_da": wind_upper_da, "pv_upper_da": pv_upper_da,
                "batt_mw_da": batt_mw_da, "tank_mw_da": tank_mw_da,
                "batt_mwh_da": batt_mwh_da, "tank_mwh_da": tank_mwh_da, "ely_mw_da": ely_mw_da}

    cache_key = (tuple(zones), rep_days_per_month, day_selection, scenario, return_duals) if reuse else None
    state = _JOINT_SUBPROBLEM_CACHE.get(cache_key) if cache_key is not None else None
    solver_kwargs = {"output_flag": False} if quiet else {}

    if state is None:
        m = linopy.Model()
        wind_p = _capacity_bounded(m, "wind_p", coords, wind_upper_da, return_duals, "wind_cap")
        pv_p = _capacity_bounded(m, "pv_p", coords, pv_upper_da, return_duals, "pv_cap")
        batt_dis = _capacity_bounded(m, "batt_dis", coords, batt_mw_da, return_duals, "batt_dis_cap")
        batt_ch = _capacity_bounded(m, "batt_ch", coords, batt_mw_da, return_duals, "batt_ch_cap")
        tank_dis = _capacity_bounded(m, "tank_dis", coords, tank_mw_da, return_duals, "tank_dis_cap")
        tank_ch = _capacity_bounded(m, "tank_ch", coords, tank_mw_da, return_duals, "tank_ch_cap")
        batt_soc = _capacity_bounded(m, "batt_soc", coords, batt_mwh_da, return_duals, "batt_soc_cap")
        tank_soc = _capacity_bounded(m, "tank_soc", coords, tank_mwh_da, return_duals, "tank_soc_cap")
        ely_p = _capacity_bounded(m, "ely_p", coords, ely_mw_da, return_duals, "ely_cap")
        variables = {"wind_p": wind_p, "pv_p": pv_p, "batt_dis": batt_dis, "batt_ch": batt_ch,
                    "tank_dis": tank_dis, "tank_ch": tank_ch, "batt_soc": batt_soc,
                    "tank_soc": tank_soc, "ely_p": ely_p}

        m.add_constraints(batt_soc.isel(hid=0) - batt_soc0_da - batt_eff * batt_ch.isel(hid=0) + batt_dis.isel(hid=0) == 0,
                          name="batt_balance_0")
        m.add_constraints(batt_soc.isel(hid=slice(1, None)) - batt_soc.isel(hid=slice(None, -1))
                          - batt_eff * batt_ch.isel(hid=slice(1, None)) + batt_dis.isel(hid=slice(1, None)) == 0,
                          name="batt_balance")
        m.add_constraints(tank_soc.isel(hid=0) - tank_soc0_da - tank_eff * tank_ch.isel(hid=0) + tank_dis.isel(hid=0) == 0,
                          name="tank_balance_0")
        m.add_constraints(tank_soc.isel(hid=slice(1, None)) - tank_soc.isel(hid=slice(None, -1))
                          - tank_eff * tank_ch.isel(hid=slice(1, None)) + tank_dis.isel(hid=slice(1, None)) == 0,
                          name="tank_balance")
        if cfg0.cyclic_storage:
            m.add_constraints(batt_soc.isel(hid=-1) >= batt_soc0_da, name="batt_cyclic")
            m.add_constraints(tank_soc.isel(hid=-1) >= tank_soc0_da, name="tank_cyclic")

        cost = (-(w * p_elec_da * (wind_p + pv_p + batt_dis - batt_ch - ely_p)).sum()
               - (w * p_h2_da * (ely_eff_da * ely_p + tank_dis - tank_ch)).sum()
               + sto_cost * ((w * batt_ch).sum() + (w * batt_dis).sum()
                            + (w * tank_ch).sum() + (w * tank_dis).sum()))
        m.add_objective(cost)

        solver = linopy_solvers.Solver.from_name(
            "highs", model=m, io_api="direct", track_updates=(cache_key is not None),
            options=solver_kwargs, set_names=False)
        build_s = time.time() - t0
        t1 = time.time()
        result = solver.solve(assign=True)
        solve_s = time.time() - t1
        if cache_key is not None:
            _JOINT_SUBPROBLEM_CACHE[cache_key] = {
                "m": m, "solver": solver, "variables": variables, "cfg0": cfg0,
            }
    else:
        m, solver, variables = state["m"], state["solver"], state["variables"]
        for name, cap_name, upper_key in _CAP_VARS:
            upper_val = upper_das[upper_key]
            if return_duals:
                m.constraints[cap_name].update(rhs=upper_val)
            else:
                variables[name].update(upper=upper_val)
        m.constraints["batt_balance_0"].update(rhs=batt_soc0_da)
        m.constraints["tank_balance_0"].update(rhs=tank_soc0_da)
        if cfg0.cyclic_storage:
            m.constraints["batt_cyclic"].update(rhs=batt_soc0_da)
            m.constraints["tank_cyclic"].update(rhs=tank_soc0_da)

        wind_p, pv_p = variables["wind_p"], variables["pv_p"]
        batt_dis, batt_ch = variables["batt_dis"], variables["batt_ch"]
        tank_dis, tank_ch, ely_p = variables["tank_dis"], variables["tank_ch"], variables["ely_p"]
        cost = (-(w * p_elec_da * (wind_p + pv_p + batt_dis - batt_ch - ely_p)).sum()
               - (w * p_h2_da * (ely_eff_da * ely_p + tank_dis - tank_ch)).sum()
               + sto_cost * ((w * batt_ch).sum() + (w * batt_dis).sum()
                            + (w * tank_ch).sum() + (w * tank_dis).sum()))
        m.add_objective(cost, overwrite=True)

        solver.update(m)
        build_s = time.time() - t0
        t1 = time.time()
        result = solver.solve(assign=True)
        solve_s = time.time() - t1

    status, condition = m.status, m.termination_condition
    if status != "ok":
        raise RuntimeError(f"joint solve failed: {status}/{condition}")

    sol = m.solution
    cut_coeffs = None
    if return_duals:
        mu_electrolyser_da = m.dual["ely_cap"].sum(["day", "hid"])
        mu_wind_da = (m.dual["wind_cap"] * wind_cf_norm_da).sum(["day", "hid"])
        mu_pv_da = (m.dual["pv_cap"] * pv_cf_norm_da).sum(["day", "hid"])
        batt_soc0_terms = m.dual["batt_balance_0"].sum("day")
        tank_soc0_terms = m.dual["tank_balance_0"].sum("day")
        if cfg0.cyclic_storage:
            batt_soc0_terms = batt_soc0_terms + m.dual["batt_cyclic"].sum("day")
            tank_soc0_terms = tank_soc0_terms + m.dual["tank_cyclic"].sum("day")
        mu_battery_da = (m.dual["batt_dis_cap"].sum(["day", "hid"]) + m.dual["batt_ch_cap"].sum(["day", "hid"])
                        + m.dual["batt_soc_cap"].sum(["day", "hid"]) * cfg0.g_investor_battery_duration_hours
                        + batt_soc0_terms * cfg0.initial_soc_fraction * cfg0.g_investor_battery_duration_hours)
        mu_tank_da = (m.dual["tank_dis_cap"].sum(["day", "hid"]) + m.dual["tank_ch_cap"].sum(["day", "hid"])
                    + m.dual["tank_soc_cap"].sum(["day", "hid"]) * cfg0.g_investor_tank_duration_hours
                    + tank_soc0_terms * cfg0.initial_soc_fraction * cfg0.g_investor_tank_duration_hours)
        cut_coeffs = {z: {"electrolyser_mw": float(mu_electrolyser_da.sel(zone=z)),
                          "wind_mw": float(mu_wind_da.sel(zone=z)),
                          "pv_mw": float(mu_pv_da.sel(zone=z)),
                          "battery_mw": float(mu_battery_da.sel(zone=z)),
                          "tank_mw": float(mu_tank_da.sel(zone=z))}
                     for z in zones}

    schedules, objective_by_zone = {}, {}
    day_weight_flat = np.repeat(np.asarray(day_weights, dtype=float), HOURS_PER_DAY)
    for zi, z in enumerate(zones):
        wp = np.asarray(sol["wind_p"].sel(zone=z).values).reshape(-1)
        pp = np.asarray(sol["pv_p"].sel(zone=z).values).reshape(-1)
        bd = np.asarray(sol["batt_dis"].sel(zone=z).values).reshape(-1)
        bc = np.asarray(sol["batt_ch"].sel(zone=z).values).reshape(-1)
        ep = np.asarray(sol["ely_p"].sel(zone=z).values).reshape(-1)
        td = np.asarray(sol["tank_dis"].sel(zone=z).values).reshape(-1)
        tc = np.asarray(sol["tank_ch"].sel(zone=z).values).reshape(-1)
        xg = wp + pp + bd - bc - ep
        xh = ely_effs[z] * ep + td - tc
        out = pd.DataFrame({
            "hour": np.arange(H),
            "H2 Producer wind (MW)": wp,
            "H2 Producer pv (MW)": pp,
            "H2 Producer battery discharge (MW)": bd,
            "H2 Producer battery charge (-) (MW)": -bc,
            "H2 Producer electrolyser load (-) (MW)": -ep,
            "H2 Producer grid exchange (MW)": xg,
            "H2 Producer electrolyser production (MW)": ely_effs[z] * ep,
            "H2 Producer tank discharge (MW)": td,
            "H2 Producer tank charge (-) (MW)": -tc,
            "H2 Producer pipeline exchange (MW)": xh,
        })
        out["day_of_year"] = np.repeat(days, HOURS_PER_DAY)
        out["day_weight"] = day_weight_flat

        p_elec_z = np.asarray(p_elecs[zi]).reshape(-1)
        p_h2_z = np.asarray(p_h2s[zi]).reshape(-1)
        z_cost = (-(p_elec_z * day_weight_flat * xg).sum() - (p_h2_z * day_weight_flat * xh).sum()
                 + sto_cost * (day_weight_flat * (bc + bd + tc + td)).sum())
        objective_by_zone[z] = float(z_cost)

        out.attrs["objective"] = float(z_cost)
        out.attrs["p_elec"] = p_elec_z
        out.attrs["p_h2"] = p_h2_z
        out.attrs["host_zone"] = host_zones[z]
        out.attrs["capacities"] = capacities[z]
        out.attrs["annualized_hours"] = annual_hours
        out.attrs["rep_days_per_month"] = rep_days_per_month
        out.attrs["sampled_days"] = days
        out.attrs["cut_coeffs"] = cut_coeffs[z] if cut_coeffs else None
        schedules[z] = out

    return {
        "objective": float(m.objective.value),
        "objective_by_zone": objective_by_zone,
        "schedules": schedules,
        "cut_coeffs": cut_coeffs,
        "build_seconds": build_s,
        "solve_seconds": solve_s,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DE00")
    ap.add_argument("--day", type=int, default=None, help="single day (shorthand for --start-day/--end-day)")
    ap.add_argument("--start-day", type=int, default=None,
                    help="contiguous-range start day, 1-364 (default 5 if no other horizon flag is given)")
    ap.add_argument("--end-day", type=int, default=None)
    ap.add_argument("--rep-days-per-month", type=int, default=None,
                    help="representative-sampling mode: N evenly-spaced days per month (1-29), "
                         "mutually exclusive with --day/--start-day/--end-day")
    args = ap.parse_args()
    if args.rep_days_per_month is not None:
        if args.day is not None or args.start_day is not None or args.end_day is not None:
            ap.error("--rep-days-per-month is mutually exclusive with --day/--start-day/--end-day")
        df = solve(args.zone, rep_days_per_month=args.rep_days_per_month)
    else:
        if args.day is not None:
            start_day, end_day = args.day, args.day
        else:
            start_day, end_day = (args.start_day or 5), (args.end_day or args.start_day or 5)
        df = solve(args.zone, start_day, end_day)
    if df.attrs["n_hours"] <= 48:
        print(df.to_string(index=False))
    else:
        print(df.describe().to_string())
    print(f"\nhours solved: {df.attrs['n_hours']} (annualized as {df.attrs['annualized_hours']}) | "
         f"build: {df.attrs['build_seconds']:.1f}s | solve: {df.attrs['solve_seconds']:.1f}s")
    print("objective (EUR, negative = net revenue):", round(df.attrs["objective"], 2))
