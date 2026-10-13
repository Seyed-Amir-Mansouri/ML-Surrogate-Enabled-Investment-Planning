"""Discrete candidate capacity grids for General Investor capacity planning."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from economic_dispatch import model as ed_model
from economic_dispatch.config import RunConfig

from .config import ASSETS, CapexAssumptions

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZONES_DB = ROOT / "inputs" / "zones_2030.parquet"
DEFAULT_NETWORKS_DB = ROOT / "inputs" / "networks_2030.parquet"


def default_sizing_and_zones(zones_db=DEFAULT_ZONES_DB, networks_db=DEFAULT_NETWORKS_DB):
    """Return (sizing, main_zones) for every country with a General Investor."""
    cfg = RunConfig(zones_db=zones_db, networks_db=networks_db)
    sizing = ed_model._g_investor_sizing(cfg)
    main_zones = ed_model._h2_main_zones(cfg)
    return sizing, main_zones


def build_candidates(countries: list[str], capex_cfg: CapexAssumptions | None = None,
                     zones_db=DEFAULT_ZONES_DB, networks_db=DEFAULT_NETWORKS_DB):
    """Build per-country candidate MW/CAPEX grids and host zones for the given countries."""
    capex_cfg = capex_cfg or CapexAssumptions()
    sizing, main_zones = default_sizing_and_zones(zones_db, networks_db)
    missing = [c for c in countries if c not in sizing]
    if missing:
        raise ValueError(f"no General Investor sizing for {missing} -- eligible "
                         f"countries: {sorted(sizing)}")

    default_mw = {c: {a: float(sizing[c][a]) for a in ASSETS} for c in countries}
    cand_mw_shared = {a: np.asarray([cand.mw for cand in capex_cfg.catalog[a]], dtype=float)
                      for a in ASSETS}
    cand_capex_shared = {a: np.asarray([cand.capex_eur for cand in capex_cfg.catalog[a]], dtype=float)
                         for a in ASSETS}
    cand_mwh_shared = {a: np.asarray([cand.mwh or 0.0 for cand in capex_cfg.catalog[a]], dtype=float)
                       for a in ASSETS}
    cand_mw = {c: dict(cand_mw_shared) for c in countries}
    cand_capex = {c: dict(cand_capex_shared) for c in countries}
    cand_mwh = {c: dict(cand_mwh_shared) for c in countries}
    host_zone = {c: main_zones[c] for c in countries}
    return default_mw, cand_mw, cand_capex, cand_mwh, host_zone
