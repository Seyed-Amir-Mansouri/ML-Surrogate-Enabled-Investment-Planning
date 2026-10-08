"""Bridge between the Django UI and the optimizer: builds plan_capacity.py commands and reads results back."""
from __future__ import annotations

import csv
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from django.conf import settings

PROJECT_ROOT = Path(settings.PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "outputs" / "webui"
SCENARIO_OVERRIDES_ENV = "PLANNER_SCENARIO_OVERRIDES"
CATALOG_OVERRIDES_ENV = "PLANNER_CATALOG_OVERRIDES"


@lru_cache(maxsize=1)
def planner_module():
    import plan_capacity

    return plan_capacity


@lru_cache(maxsize=1)
def capex_assumptions_defaults():
    import g_investor_planning as hp

    cfg = hp.CapexAssumptions()
    return {
        "assets": list(hp.ASSETS),
        "catalog": {a: [c._asdict() for c in cfg.catalog[a]] for a in hp.ASSETS},
        "discount_rate": cfg.discount_rate,
        "budget": cfg.default_budget_eur,
        "lifetime_years": dict(cfg.lifetime_years),
    }


@lru_cache(maxsize=1)
def eligible_countries() -> list[str]:
    return planner_module().eligible_countries()


ASSET_MAP_TOKENS = {
    "electrolyser_mw": "--asset-electrolyser",
    "wind_mw": "--asset-wind",
    "pv_mw": "--asset-pv",
    "battery_mw": "--asset-battery",
    "tank_mw": "--asset-tank",
}
MARKER_MIN_DIAMETER = 18
MARKER_MAX_DIAMETER = 56


def _marker_diameter(total: float, max_total: float) -> float:
    if max_total <= 0:
        return MARKER_MIN_DIAMETER
    scale = (total / max_total) ** 0.5
    return round(MARKER_MIN_DIAMETER + (MARKER_MAX_DIAMETER - MARKER_MIN_DIAMETER) * scale, 1)


def _marker_gradient(values: dict[str, float], total: float, assets: list[str]) -> str:
    if total <= 0:
        return ""
    stops = []
    acc = 0.0
    for asset in assets:
        share = values.get(asset, 0.0)
        if share <= 0:
            continue
        pct = share / total * 100
        stops.append(f"var({ASSET_MAP_TOKENS[asset]}) {acc:.2f}% {acc + pct:.2f}%")
        acc += pct
    return "conic-gradient(" + ", ".join(stops) + ")"


def country_map_markers(params: dict, summary: dict) -> list[dict]:
    from . import geo

    assets = capex_assumptions_defaults()["assets"]
    by_country = {row["country"]: row for row in summary.get("capacities", [])}
    codes = eligible_countries() if params.get("all_countries") else params.get("countries", [])

    rows = []
    for code in codes:
        centroid = geo.COUNTRY_CENTROIDS.get(code)
        if centroid is None:
            continue
        row = by_country.get(code, {})
        values = {a: float(row.get(a, 0) or 0) for a in assets}
        rows.append({
            "code": code,
            "lat": centroid[0],
            "lon": centroid[1],
            "total": sum(values.values()),
            "values": values,
        })

    max_total = max((r["total"] for r in rows), default=0.0)
    for r in rows:
        r["diameter"] = _marker_diameter(r["total"], max_total)
        r["gradient"] = _marker_gradient(r["values"], r["total"], assets)
    return rows


def economics(params: dict, summary: dict) -> dict:
    import g_investor_planning as hp

    cfg = hp.CapexAssumptions(discount_rate=params.get("discount_rate_pct", 5) / 100)
    if params.get("lifetime_years"):
        cfg.lifetime_years = {a: params["lifetime_years"] for a in hp.ASSETS}
    crfs = cfg.capital_recovery_factors()

    capex_by_asset = {a: 0.0 for a in hp.ASSETS}
    for row in summary.get("units", []):
        asset = row.get("asset")
        if asset not in capex_by_asset:
            continue
        candidate = cfg.catalog[asset][int(row["candidate"])]
        capex_by_asset[asset] += candidate.capex_eur * float(row["units"])

    annualized_by_asset = {a: capex_by_asset[a] * crfs[a] for a in hp.ASSETS}
    annualized_total = sum(annualized_by_asset.values())
    raw_capex = summary.get("raw_capex_eur")
    if raw_capex is None:
        raw_capex = sum(capex_by_asset.values())
    objective = summary.get("objective_eur")
    total_mw = sum(summary.get("totals_mw", {}).values())

    return {
        "raw_capex_eur": raw_capex,
        "annualized_capex_eur": annualized_total,
        "operating_cost_eur": (objective - annualized_total) if objective is not None else None,
        "capex_per_mw_eur": (raw_capex / total_mw) if total_mw else None,
        "by_asset": {
            a: {"capex_eur": capex_by_asset[a], "annualized_eur": annualized_by_asset[a], "crf": crfs[a]}
            for a in hp.ASSETS
        },
    }


@lru_cache(maxsize=1)
def scenario_probabilities() -> dict[str, float]:
    return dict(planner_module().SCENARIO_PROBS)


@lru_cache(maxsize=1)
def scenario_defaults() -> dict[str, dict]:
    path = PROJECT_ROOT / "inputs" / "uncertainty_scenarios.json"
    return json.loads(path.read_text(encoding="utf-8"))["scenarios"]


def build_command(params: dict, output_prefix: Path) -> list[str]:
    cmd = [sys.executable, "plan_capacity.py"]
    if params["all_countries"]:
        cmd.append("--all")
    else:
        cmd += ["--countries", ",".join(params["countries"])]
    cmd += ["--budget", f"{params['budget']:.0f}",
            "--max-units-per-candidate", str(params["max_units_per_candidate"]),
            "--discount-rate", f"{params['discount_rate_pct'] / 100:.6f}",
            "--rep-days-per-month", str(params["rep_days_per_month"]),
            "--gap-tol", str(params["gap_tol"]),
            "--max-iters", str(params["max_iters"]),
            "--master-time-limit", str(params["master_time_limit"]),
            "--workers", str(params["workers"]),
            "--cvar-alpha", str(params["cvar_alpha"]) if params["risk_measure"] == "cvar" else "off",
            "--output", str(output_prefix)]
    if params.get("lifetime_years"):
        cmd += ["--lifetime-years", str(params["lifetime_years"])]
    if params.get("disabled_assets"):
        cmd += ["--disabled-assets", ",".join(params["disabled_assets"])]
    if params.get("scenarios") and set(params["scenarios"]) != set(scenario_probabilities()):
        cmd += ["--scenarios", ",".join(params["scenarios"])]
    return cmd


def run_dir(run_pk: int) -> Path:
    return RUNS_DIR / f"run_{run_pk}"


def output_prefix_for(run_pk: int) -> Path:
    return run_dir(run_pk) / "plan"


_DONE_RE = re.compile(r"Done in ([\d.]+)s")
_CAPEX_RE = re.compile(r"Total raw CAPEX: ([\d,]+) EUR")
_OBJ_RE = re.compile(r"Best objective.*?: (-?[\d,]+)\s*$", re.MULTILINE)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def parse_summary(output_prefix: Path, log: str) -> dict:
    capacities = _read_csv(Path(f"{output_prefix}_capacities.csv"))
    units = _read_csv(Path(f"{output_prefix}_units.csv"))
    convergence = _read_csv(Path(f"{output_prefix}_convergence.csv"))

    assets = list(capex_assumptions_defaults()["assets"])
    totals = {a: 0.0 for a in assets}
    for row in capacities:
        for a in assets:
            totals[a] += float(row[a])

    done = _DONE_RE.search(log)
    capex = _CAPEX_RE.search(log)
    objective = _OBJ_RE.search(log)
    return {
        "elapsed_seconds": float(done.group(1)) if done else None,
        "iterations": len(convergence),
        "converged": bool(re.search(r"converged \(gap", log)),
        "objective_eur": _num(objective.group(1)) if objective else None,
        "raw_capex_eur": _num(capex.group(1)) if capex else None,
        "capacities": capacities,
        "units": units,
        "convergence": [{k: _maybe_float(v) for k, v in row.items()} for row in convergence],
        "totals_mw": totals,
    }


def _maybe_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value
