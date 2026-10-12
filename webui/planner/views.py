from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from . import geo, runner, services
from .forms import ASSET_LABELS, PlanRunForm
from .models import PlanRun


def _running_count() -> int:
    return PlanRun.objects.filter(status=PlanRun.Status.RUNNING).count()


def index(request):
    runs = PlanRun.objects.all()[:100]
    return render(request, "planner/index.html", {
        "runs": runs,
        "running_now": _running_count() > 0,
        "asset_labels": ASSET_LABELS,
    })


def new_run(request):
    if request.method == "POST":
        form = PlanRunForm(request.POST)
        if form.is_valid():
            if _running_count() >= settings.MAX_CONCURRENT_RUNS:
                messages.error(request, "A run is already in progress. Wait for it to finish, then start this one.")
            else:
                with transaction.atomic():
                    run = PlanRun.objects.create(name=form.cleaned_data["name"], params=form.params(),
                                                 output_prefix="")
                    run.output_prefix = str(services.output_prefix_for(run.pk))
                    run.save(update_fields=["output_prefix"])
                runner.start(run)
                return redirect(reverse("run_detail", args=[run.pk]))
    else:
        form = PlanRunForm()
    return render(request, "planner/new.html", {"form": form, "running_now": _running_count() > 0})


def run_detail(request, pk: int):
    run = get_object_or_404(PlanRun, pk=pk)
    summary = run.summary or {}
    return render(request, "planner/detail.html", {
        "run": run,
        "summary": summary,
        "convergence": summary.get("convergence", []),
        "asset_labels": ASSET_LABELS,
        "assets": list(ASSET_LABELS),
        "map_markers": services.country_map_markers(run.params, summary) if summary else [],
        "asset_tokens": services.ASSET_MAP_TOKENS,
        "map_center": geo.DEFAULT_CENTER,
        "map_zoom": geo.DEFAULT_ZOOM,
        "economics": services.economics(run.params, summary) if summary else {},
        "log_tail": "\n".join(run.log.strip().splitlines()[-60:]) if run.log else "",
    })


def compare(request):
    done = PlanRun.objects.filter(status=PlanRun.Status.DONE)
    a_id, b_id = request.GET.get("a"), request.GET.get("b")
    run_a = done.filter(pk=a_id).first() if a_id else None
    run_b = done.filter(pk=b_id).first() if b_id else None

    rows = []
    econ_a, econ_b, econ_rows = {}, {}, []
    if run_a and run_b:
        caps_a = {r["country"]: r for r in run_a.summary.get("capacities", [])}
        caps_b = {r["country"]: r for r in run_b.summary.get("capacities", [])}
        for country in sorted(set(caps_a) | set(caps_b)):
            ra, rb = caps_a.get(country, {}), caps_b.get(country, {})
            mw_a = sum(float(ra.get(a, 0) or 0) for a in ASSET_LABELS)
            mw_b = sum(float(rb.get(a, 0) or 0) for a in ASSET_LABELS)
            rows.append({"country": country, "mw_a": mw_a, "mw_b": mw_b, "delta": mw_b - mw_a})

        econ_a = services.economics(run_a.params, run_a.summary)
        econ_b = services.economics(run_b.params, run_b.summary)
        for a in ASSET_LABELS:
            ea, eb = econ_a["by_asset"][a], econ_b["by_asset"][a]
            econ_rows.append({
                "asset": a,
                "annualized_a": ea["annualized_eur"],
                "annualized_b": eb["annualized_eur"],
                "delta": eb["annualized_eur"] - ea["annualized_eur"],
            })

    return render(request, "planner/compare.html", {
        "done_runs": done[:100],
        "run_a": run_a,
        "run_b": run_b,
        "rows": rows,
        "asset_labels": ASSET_LABELS,
        "econ_a": econ_a,
        "econ_b": econ_b,
        "econ_rows": econ_rows,
    })


def catalog(request):
    return render(request, "planner/catalog.html", {
        "defaults": services.capex_assumptions_defaults(),
        "asset_labels": ASSET_LABELS,
    })
