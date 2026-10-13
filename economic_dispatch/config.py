"""Run configuration and tunable assumptions for the dispatch model."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ALL_ZONES = [
    "AT00", "BE00", "BEOF", "CZ00", "DE00", "DEKF", "FR00", "FR15", "HR00",
    "HU00", "LUB1", "LUF1", "LUG1", "LUV1", "NL00", "NLLL", "PL00",
    "RO00", "SI00", "SK00",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_EXPORTS_DIR = PROJECT_ROOT / "inputs"
DEFAULT_ZONES_DB = DEFAULT_EXPORTS_DIR / "zones_2030.parquet"
DEFAULT_NETWORKS_DB = DEFAULT_EXPORTS_DIR / "networks_2030.parquet"
DEFAULT_MARGINAL_PRICE_ELEC_DB = DEFAULT_EXPORTS_DIR / "marginal_price_electricity_2030.parquet"
DEFAULT_MARGINAL_PRICE_H2_DB = DEFAULT_EXPORTS_DIR / "marginal_price_hydrogen_2030.parquet"

HOURS_PER_DAY = 24
HOURS_PER_YEAR = 8736


@lru_cache(maxsize=8)
def discover_zones(zones_db=DEFAULT_ZONES_DB) -> list[str]:
    import pandas as pd
    return sorted(pd.read_parquet(Path(zones_db), columns=["zone"])["zone"].unique().tolist())


def _expand_to_countries(zones: list[str], zones_db) -> list[str]:
    all_zones = discover_zones(zones_db)
    countries = {z[:2] for z in zones}
    return sorted(set(zones) | {z for z in all_zones if z[:2] in countries})


@dataclass
class RunConfig:
    zones: list[str] = field(default_factory=lambda: list(ALL_ZONES))
    start_day: int = 1
    end_day: int = 1
    output_dir: Path = DEFAULT_OUTPUT_DIR
    exports_dir: Path = DEFAULT_EXPORTS_DIR
    zones_db: Path = DEFAULT_ZONES_DB
    networks_db: Path = DEFAULT_NETWORKS_DB
    out_tag: str | None = None

    enable_h2_storage: bool = True
    cyclic_storage: bool = True
    enable_uc: bool = False
    subtract_dsr_implicit: bool = False
    electricity_only: bool = False

    fuel_per_thermal: bool = True
    co2_per_thermal: bool = True
    default_efficiency: float = 0.5
    voll_eur_per_mwh: float = 3_000.0
    h2_terminal_price: float = 150.0
    dump_penalty_eur_per_mwh: float = 0.0
    storage_op_cost_eur_per_mwh: float = 0.01

    initial_soc_fraction: float = 0.5
    ramp_scale: float = 1.0
    default_pump_efficiency: float = 0.8
    default_closed_ps_efficiency: float = 0.75
    h2_storage_hours: float = 168.0
    h2_storage_efficiency: float = 1.0
    default_hydro_efficiency: float = 1.0

    g_investor_renewable_pct_of_electrolyser_mw: float = 0.30
    g_investor_wind_to_pv_ratio: float = 1.3
    g_investor_renewable_capacity_step_mw: float = 2.5
    g_investor_electrolyser_efficiency: float = 0.68
    g_investor_battery_pct_of_electrolyser_mw: float = 0.25
    g_investor_battery_duration_hours: float = 2.0
    g_investor_tank_pct_of_electrolyser_h2: float = 0.50
    g_investor_tank_duration_hours: float = 24.0
    g_investor_battery_tank_step_mw: float = 2.5
    g_investor_battery_efficiency: float = 0.92
    g_investor_tank_efficiency: float = 1.0
    g_investor_electrolyser_capacities_mw: list[float] = field(
        default_factory=lambda: [5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 30, 35, 40])

    g_investor_electrolyser_mw_overrides: dict[str, float] = field(default_factory=dict)
    g_investor_wind_mw_overrides: dict[str, float] = field(default_factory=dict)
    g_investor_pv_mw_overrides: dict[str, float] = field(default_factory=dict)
    g_investor_battery_mw_overrides: dict[str, float] = field(default_factory=dict)
    g_investor_tank_mw_overrides: dict[str, float] = field(default_factory=dict)
    # Energy capacity (MWh) for an overridden battery/tank MW, e.g. from the actual mix of
    # candidate products chosen by the capacity planner. Falls back to mw_override *
    # g_investor_{battery,tank}_duration_hours when a country has no entry here.
    g_investor_battery_mwh_overrides: dict[str, float] = field(default_factory=dict)
    g_investor_tank_mwh_overrides: dict[str, float] = field(default_factory=dict)

    solver_name: str = "highs"
    mip_rel_gap: float = 1e-4

    capacity_scale: dict[str, float] = field(default_factory=dict)
    use_plexos_renewable_override: bool = True

    external_import_leg_cap: dict | None = None
    external_export_leg_cap: dict | None = None

    def __post_init__(self) -> None:
        self.zones = _expand_to_countries(self.zones, self.zones_db)

    def resolved_output_dir(self) -> Path:
        """Output folder for this run: outputs/ or outputs/<out_tag>/ if tagged."""
        base = Path(self.output_dir)
        return base / self.out_tag if self.out_tag else base

    def hour_slice(self) -> tuple[int, int]:
        """0-based half-open (start_row, end_row) into the 8736-hour year for [start_day, end_day]."""
        start = (self.start_day - 1) * HOURS_PER_DAY
        end = self.end_day * HOURS_PER_DAY
        return start, end

    def num_days(self) -> int:
        return self.end_day - self.start_day + 1

    def month_index(self) -> int:
        """Approx calendar month (0-based) of the first day, for must-run selection."""
        day0 = self.start_day - 1
        return min(11, int(day0 / (364 / 12)))
