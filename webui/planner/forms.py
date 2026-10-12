from django import forms

from . import services
from .geo import country_label

ASSET_LABELS = {
    "electrolyser_mw": "Electrolyser",
    "wind_mw": "Wind",
    "pv_mw": "Solar PV",
    "battery_mw": "Battery",
    "tank_mw": "H2 tank",
}
MWH_ASSETS = {"battery_mw", "tank_mw"}
MAX_CANDIDATES_PER_ASSET = 8
BASELINE_SCENARIO = "p100"
DEFAULT_ERROR_RANGE = (-100.0, 100.0)


class PlanRunForm(forms.Form):
    # Problem definition
    name = forms.CharField(max_length=120, required=False, label="Run name",
                           widget=forms.TextInput(attrs={"placeholder": "e.g. Baseline 500M, PV only"}))
    all_countries = forms.BooleanField(required=False, initial=True, label="All eligible countries")
    countries = forms.MultipleChoiceField(required=False, label="Countries",
                                          widget=forms.CheckboxSelectMultiple)
    budget = forms.FloatField(min_value=1e6, initial=500_000_000, label="Total CAPEX budget (EUR)",
                              help_text="Raw, unannualized budget across all countries.")
    active_assets = forms.MultipleChoiceField(
        required=False, label="Active asset types", initial=list(ASSET_LABELS),
        choices=[(a, label) for a, label in ASSET_LABELS.items()],
        widget=forms.CheckboxSelectMultiple,
    )

    # Economics
    discount_rate_pct = forms.FloatField(min_value=0, max_value=30, initial=5, label="Discount rate (%)")
    cvar_alpha = forms.FloatField(required=False, min_value=0.5, max_value=0.99, initial=0.8,
                                  label="CVaR confidence level α",
                                  help_text="Higher α focuses on the worst tail of scenarios.")

    # Solver
    rep_days_per_month = forms.IntegerField(min_value=1, max_value=29, initial=7,
                                            label="Representative days per month",
                                            help_text="More days = more accurate but slower.")
    gap_tol = forms.FloatField(min_value=0.0001, max_value=0.5, initial=0.01, label="Optimality gap tolerance")
    max_iters = forms.IntegerField(min_value=1, max_value=100, initial=100, label="Max Benders iterations")
    master_time_limit = forms.FloatField(min_value=10, max_value=3600, initial=180,
                                         label="Master time limit (s)")
    workers = forms.IntegerField(min_value=1, max_value=8, initial=2, label="Parallel workers",
                                 help_text="Workers solve each iteration's scenario subproblems in parallel. "
                                           "Each one uses significant memory, so use 2 or fewer on constrained machines.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.defaults = services.scenario_defaults()
        self.scenario_countries = sorted(next(iter(self.defaults.values()))["wind"])
        self.fields["countries"].choices = [(c, country_label(c)) for c in services.eligible_countries()]

        self.catalog_defaults = services.capex_assumptions_defaults()
        self.assets = self.catalog_defaults["assets"]
        for asset in self.assets:
            rows = self.catalog_defaults["catalog"][asset]
            lifetime_default = rows[0]["lifetime_years"] if rows else 1.0
            self.fields[f"cat_lifetime__{asset}"] = forms.FloatField(
                min_value=0.1, max_value=100, initial=lifetime_default, label="Lifetime (yr)")
            for i in range(MAX_CANDIDATES_PER_ASSET):
                c = rows[i] if i < len(rows) else None
                self.fields[f"cat_mw__{asset}__{i}"] = forms.FloatField(
                    required=False, min_value=0.001, initial=c["mw"] if c else None, label="Size (MW)")
                self.fields[f"cat_capex__{asset}__{i}"] = forms.FloatField(
                    required=False, min_value=0, initial=c["capex_eur"] if c else None, label="CAPEX (EUR)")
                if asset in MWH_ASSETS:
                    self.fields[f"cat_mwh__{asset}__{i}"] = forms.FloatField(
                        required=False, min_value=0, initial=c["mwh"] if c else None, label="Energy (MWh)")

        # Per-country, per-asset accepted error % range the price models were trained on (see
        # services.scenario_capacity_ranges). common_error_range is the intersection across every
        # country, so filling that range into every country's box can never push one out of range.
        self.capacity_ranges = services.scenario_capacity_ranges()
        self.common_error_range = {
            asset: (
                max(self._error_range(c, asset)[0] for c in self.scenario_countries),
                min(self._error_range(c, asset)[1] for c in self.scenario_countries),
            )
            for asset in ("wind", "solar")
        }

        # Baseline (p100, zero wind/solar error) is the only scenario with real pre-researched
        # content shown to the user. The other scenario keys in uncertainty_scenarios.json
        # (unc01..unc10) are reused purely as blank internal slots -- the override mechanism
        # (PLANNER_SCENARIO_OVERRIDES) can only tweak an *existing* scenario key, not introduce a
        # new one, so "custom" scenarios borrow their key/identity but none of their JSON content.
        self.custom_scenario_keys = sorted(s for s in self.defaults if s != BASELINE_SCENARIO)
        for s in self.defaults:
            is_baseline = s == BASELINE_SCENARIO
            self.fields[f"scenario_include__{s}"] = forms.BooleanField(required=False, initial=is_baseline)
            self.fields[f"scenario_prob__{s}"] = forms.FloatField(
                min_value=0, max_value=100, initial=(100 if is_baseline else 0), label="Probability (%)")
            # Baseline is always "as planned" (no change) -- it has no change % fields at all, so it
            # can't be edited even via a crafted POST. params() hardcodes its wind/solar scale to 1.0.
            if is_baseline:
                continue
            for c in self.scenario_countries:
                for asset, field_prefix in (("wind", "err_wind"), ("solar", "err_solar")):
                    low, high = self._error_range(c, asset)
                    self.fields[f"{field_prefix}__{s}__{c}"] = forms.FloatField(
                        initial=0, label=f"{c} {asset} change (%)",
                        min_value=low, max_value=high,
                        widget=forms.NumberInput(attrs={"min": low, "max": high, "step": "0.1"}),
                        error_messages={
                            # Django %-formats these against the validator's params even when the
                            # message has no placeholder, so a literal "%" must be escaped as "%%".
                            "min_value": f"Outside the accepted range ({low:+.0f} to {high:+.0f}%%).",
                            "max_value": f"Outside the accepted range ({low:+.0f} to {high:+.0f}%%).",
                        },
                    )
        for i, s in enumerate(self.custom_scenario_keys):
            self.fields[f"scenario_name__{s}"] = forms.CharField(
                required=False, max_length=60, initial=f"Scenario {i + 2}", label="Scenario name",
                widget=forms.TextInput(attrs={"class": "scenario-name-input", "placeholder": "Scenario name"}))

    def _error_range(self, country: str, asset: str) -> tuple[float, float]:
        return self.capacity_ranges.get(country, {}).get(asset, DEFAULT_ERROR_RANGE)

    @property
    def has_capacity_ranges(self) -> bool:
        return bool(self.capacity_ranges)

    @staticmethod
    def _range_text(low: float, high: float) -> str:
        return f"{low:+.0f} to {high:+.0f}%"

    COUNTRY_TABLE_COLUMNS = 4
    ASSET_TABLE_COLUMNS = 5

    @property
    def country_table_rows(self) -> list[list]:
        cells = list(self["countries"])
        cols = self.COUNTRY_TABLE_COLUMNS
        return [cells[i:i + cols] for i in range(0, len(cells), cols)]

    @property
    def asset_table_rows(self) -> list[list]:
        cells = list(self["active_assets"])
        cols = self.ASSET_TABLE_COLUMNS
        return [cells[i:i + cols] for i in range(0, len(cells), cols)]

    @property
    def catalog_rows(self) -> list[dict]:
        rows = []
        for asset in self.assets:
            rows.append({
                "asset": asset,
                "label": ASSET_LABELS[asset],
                "has_mwh": asset in MWH_ASSETS,
                "lifetime": self[f"cat_lifetime__{asset}"],
                "candidates": [
                    {
                        "index": i,
                        "mw": self[f"cat_mw__{asset}__{i}"],
                        "capex": self[f"cat_capex__{asset}__{i}"],
                        "mwh": self[f"cat_mwh__{asset}__{i}"] if asset in MWH_ASSETS else None,
                    }
                    for i in range(MAX_CANDIDATES_PER_ASSET)
                ],
            })
        return rows

    @property
    def active_scenario_count(self) -> int:
        return sum(1 for s in self.defaults if self[f"scenario_include__{s}"].value())

    @property
    def risk_measure_preview(self) -> str:
        return "deterministic" if self.active_scenario_count == 1 else "cvar"

    @property
    def common_range_text(self) -> dict[str, str]:
        return {asset: self._range_text(*self.common_error_range[asset]) for asset in ("wind", "solar")}

    @property
    def scenario_rows(self) -> list[dict]:
        def row(s, is_baseline):
            countries = []
            if not is_baseline:
                countries = [{"code": c,
                              "wind": self[f"err_wind__{s}__{c}"],
                              "solar": self[f"err_solar__{s}__{c}"],
                              "wind_range": self._range_text(*self._error_range(c, "wind")),
                              "solar_range": self._range_text(*self._error_range(c, "solar"))}
                             for c in self.scenario_countries]
            return {
                "name": s,
                "is_baseline": is_baseline,
                "name_field": None if is_baseline else self[f"scenario_name__{s}"],
                "include": self[f"scenario_include__{s}"],
                "prob": self[f"scenario_prob__{s}"],
                "countries": countries,
            }
        return [row(BASELINE_SCENARIO, True)] + [row(s, False) for s in self.custom_scenario_keys]

    def clean(self):
        data = super().clean()
        if not data.get("all_countries") and not data.get("countries"):
            self.add_error("countries", "Select at least one country or choose all countries.")
        included = [s for s in self.defaults if data.get(f"scenario_include__{s}")]
        if not included:
            self.add_error(None, "Select at least one uncertainty scenario.")
        probs = [data.get(f"scenario_prob__{s}") for s in included]
        if all(p is not None for p in probs):
            total = sum(probs)
            if abs(total - 100) > 0.01:
                self.add_error(None, f"Scenario probabilities add up to {total:.2f}%. They must add up to 100%.")
        if len(included) > 1 and data.get("cvar_alpha") is None:
            self.add_error("cvar_alpha", "Enter a confidence level for CVaR.")

        for asset in self.assets:
            count = 0
            for i in range(MAX_CANDIDATES_PER_ASSET):
                mw = data.get(f"cat_mw__{asset}__{i}")
                capex = data.get(f"cat_capex__{asset}__{i}")
                mwh = data.get(f"cat_mwh__{asset}__{i}") if asset in MWH_ASSETS else None
                if mw is None and capex is None and mwh is None:
                    continue
                if mw is None or capex is None:
                    self.add_error(f"cat_mw__{asset}__{i}",
                                   "Enter both size (MW) and CAPEX for this candidate, or leave the row blank.")
                    continue
                if asset in MWH_ASSETS and mwh is None:
                    self.add_error(f"cat_mwh__{asset}__{i}",
                                   "Enter the energy (MWh) for this candidate, or leave the row blank.")
                    continue
                count += 1
            if count == 0:
                self.add_error(None, f"{ASSET_LABELS[asset]}: enter at least one candidate product (size + CAPEX).")
        return data

    def params(self) -> dict:
        d = self.cleaned_data
        catalog_overrides = {}
        for asset in self.assets:
            candidates = []
            for i in range(MAX_CANDIDATES_PER_ASSET):
                mw = d.get(f"cat_mw__{asset}__{i}")
                capex = d.get(f"cat_capex__{asset}__{i}")
                if mw is None or capex is None:
                    continue
                candidate = {"mw": mw, "capex_eur": capex, "lifetime_years": d[f"cat_lifetime__{asset}"]}
                if asset in MWH_ASSETS:
                    candidate["mwh"] = d.get(f"cat_mwh__{asset}__{i}")
                candidates.append(candidate)
            catalog_overrides[asset] = sorted(candidates, key=lambda c: c["mw"])

        included = [s for s in self.defaults if d.get(f"scenario_include__{s}")]
        risk_measure = "deterministic" if len(included) == 1 else "cvar"
        total = sum(d[f"scenario_prob__{s}"] for s in included)
        overrides = {}
        for s in self.defaults:
            probability = (d[f"scenario_prob__{s}"] / total) if s in included else 0.0
            if s == BASELINE_SCENARIO:
                wind = {c: 1.0 for c in self.scenario_countries}
                solar = {c: 1.0 for c in self.scenario_countries}
            else:
                wind = {c: round(1 + d[f"err_wind__{s}__{c}"] / 100, 6) for c in self.scenario_countries}
                solar = {c: round(1 + d[f"err_solar__{s}__{c}"] / 100, 6) for c in self.scenario_countries}
            overrides[s] = {"probability": probability, "wind": wind, "solar": solar}
            if s in self.custom_scenario_keys:
                overrides[s]["label"] = d.get(f"scenario_name__{s}") or s
        active = set(d.get("active_assets", []))
        disabled_assets = [a for a in self.assets if a not in active]
        return {
            "all_countries": d["all_countries"],
            "countries": sorted(d["countries"]),
            "budget": d["budget"],
            "disabled_assets": disabled_assets,
            "scenarios": included,
            "scenario_overrides": overrides,
            "catalog_overrides": catalog_overrides,
            "discount_rate_pct": d["discount_rate_pct"],
            "risk_measure": risk_measure,
            "cvar_alpha": d["cvar_alpha"] if risk_measure == "cvar" else None,
            "rep_days_per_month": d["rep_days_per_month"],
            "gap_tol": d["gap_tol"],
            "max_iters": d["max_iters"],
            "master_time_limit": d["master_time_limit"],
            "workers": d["workers"],
        }
