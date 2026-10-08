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


class PlanRunForm(forms.Form):
    # Problem definition
    name = forms.CharField(max_length=120, required=False, label="Run name",
                           widget=forms.TextInput(attrs={"placeholder": "e.g. Baseline 500M, PV only"}))
    all_countries = forms.BooleanField(required=False, initial=True, label="All eligible countries")
    countries = forms.MultipleChoiceField(required=False, label="Countries",
                                          widget=forms.CheckboxSelectMultiple)
    budget = forms.FloatField(min_value=1e6, initial=500_000_000, label="Total CAPEX budget (EUR)",
                              help_text="Raw, unannualized budget across all countries.")
    disabled_assets = forms.MultipleChoiceField(
        required=False, label="Excluded asset types",
        choices=[(a, label) for a, label in ASSET_LABELS.items()],
        widget=forms.CheckboxSelectMultiple,
    )
    max_units_per_candidate = forms.IntegerField(
        min_value=0, initial=0, label="Max units per product",
        help_text="Cap on how many units of any single product size can be built. 0 = no cap.")

    # Economics
    discount_rate_pct = forms.FloatField(min_value=0, max_value=30, initial=5, label="Discount rate (%)")
    risk_measure = forms.ChoiceField(
        choices=[("cvar", "CVaR (risk-averse)"), ("expected", "Expected value (risk-neutral)")],
        initial="cvar", label="Risk measure", widget=forms.RadioSelect)
    cvar_alpha = forms.FloatField(required=False, min_value=0.5, max_value=0.99, initial=0.8,
                                  label="CVaR confidence level α",
                                  help_text="Higher α focuses on the worst tail of scenarios.")

    # Solver
    rep_days_per_month = forms.IntegerField(min_value=1, max_value=29, initial=7,
                                            label="Representative days per month",
                                            help_text="More days = more accurate but slower.")
    gap_tol = forms.FloatField(min_value=0.0001, max_value=0.5, initial=0.01, label="Optimality gap tolerance")
    max_iters = forms.IntegerField(min_value=1, max_value=100, initial=30, label="Max Benders iterations")
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

        for s, d in self.defaults.items():
            self.fields[f"scenario_include__{s}"] = forms.BooleanField(required=False, initial=True)
            self.fields[f"scenario_prob__{s}"] = forms.FloatField(
                min_value=0, max_value=100, initial=round(d["probability"] * 100, 4), label="Probability (%)")
            for c in self.scenario_countries:
                self.fields[f"err_wind__{s}__{c}"] = forms.FloatField(
                    min_value=0, max_value=100, initial=round((1 - d["wind"].get(c, 1.0)) * 100, 4),
                    label=f"{c} wind error (%)")
                self.fields[f"err_solar__{s}__{c}"] = forms.FloatField(
                    min_value=0, max_value=100, initial=round((1 - d["solar"].get(c, 1.0)) * 100, 4),
                    label=f"{c} solar error (%)")

    COUNTRY_TABLE_COLUMNS = 4

    @property
    def country_table_rows(self) -> list[list]:
        cells = list(self["countries"])
        cols = self.COUNTRY_TABLE_COLUMNS
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
    def scenario_rows(self) -> list[dict]:
        rows = []
        for s, d in self.defaults.items():
            rows.append({
                "name": s,
                "band": d.get("systemic_band", ""),
                "protected": d.get("protected_country", ""),
                "sys_wind": d.get("systemic_wind_reduction_pct"),
                "sys_solar": d.get("systemic_solar_reduction_pct"),
                "include": self[f"scenario_include__{s}"],
                "prob": self[f"scenario_prob__{s}"],
                "countries": [{"code": c,
                               "wind": self[f"err_wind__{s}__{c}"],
                               "solar": self[f"err_solar__{s}__{c}"]}
                              for c in self.scenario_countries],
            })
        return rows

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
        if data.get("risk_measure") == "cvar" and data.get("cvar_alpha") is None:
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
        total = sum(d[f"scenario_prob__{s}"] for s in included)
        overrides = {}
        for s in self.defaults:
            probability = (d[f"scenario_prob__{s}"] / total) if s in included else 0.0
            overrides[s] = {
                "probability": probability,
                "wind": {c: round(1 - d[f"err_wind__{s}__{c}"] / 100, 6) for c in self.scenario_countries},
                "solar": {c: round(1 - d[f"err_solar__{s}__{c}"] / 100, 6) for c in self.scenario_countries},
            }
        return {
            "all_countries": d["all_countries"],
            "countries": sorted(d["countries"]),
            "budget": d["budget"],
            "disabled_assets": list(d["disabled_assets"]),
            "max_units_per_candidate": d["max_units_per_candidate"],
            "scenarios": included,
            "scenario_overrides": overrides,
            "catalog_overrides": catalog_overrides,
            "discount_rate_pct": d["discount_rate_pct"],
            "risk_measure": d["risk_measure"],
            "cvar_alpha": d["cvar_alpha"],
            "rep_days_per_month": d["rep_days_per_month"],
            "gap_tol": d["gap_tol"],
            "max_iters": d["max_iters"],
            "master_time_limit": d["master_time_limit"],
            "workers": d["workers"],
        }
