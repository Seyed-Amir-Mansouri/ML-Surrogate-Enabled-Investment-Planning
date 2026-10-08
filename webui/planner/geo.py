"""Country centroids for the results map (approximate, decimal degrees)."""
from __future__ import annotations

from django.utils.html import format_html, mark_safe

COUNTRY_NAMES: dict[str, str] = {
    "AT": "Austria",
    "BE": "Belgium",
    "CZ": "Czechia",
    "DE": "Germany",
    "FR": "France",
    "HR": "Croatia",
    "HU": "Hungary",
    "LU": "Luxembourg",
    "NL": "Netherlands",
    "PL": "Poland",
    "RO": "Romania",
    "SI": "Slovenia",
    "SK": "Slovakia",
}

# Simplified flag artwork (stripes only, no emblems) as <rect> fills on a 0 0 4 3 viewBox.
# Windows doesn't render Unicode regional-indicator emoji as flag pictures (it shows the
# bare letters instead), so real vector art is used rather than the "\U0001F1E6..." emoji trick.
_H = "horizontal"
_V = "vertical"
COUNTRY_FLAG_STRIPES: dict[str, tuple[str, list[str]]] = {
    "AT": (_H, ["#ED2939", "#fff", "#ED2939"]),
    "BE": (_V, ["#000", "#FDDA24", "#EF3340"]),
    "CZ": (_H, ["#fff", "#D7141A"]),  # blue hoist triangle added separately
    "DE": (_H, ["#000", "#DD0000", "#FFCE00"]),
    "FR": (_V, ["#0055A4", "#fff", "#EF4135"]),
    "HR": (_H, ["#FF0000", "#fff", "#171796"]),
    "HU": (_H, ["#CD2A3E", "#fff", "#436F4D"]),
    "LU": (_H, ["#ED2939", "#fff", "#00A1DE"]),
    "NL": (_H, ["#AE1C28", "#fff", "#21468B"]),
    "PL": (_H, ["#fff", "#DC143C"]),
    "RO": (_V, ["#002B7F", "#FCD116", "#CE1126"]),
    "SI": (_H, ["#fff", "#0043A6", "#D7141A"]),
    "SK": (_H, ["#fff", "#0B4EA2", "#EE1C25"]),
}


def country_flag_svg(code: str) -> str:
    """Small inline SVG flag (stripes-only approximation) for an ISO 3166-1 alpha-2 code."""
    code = code.upper()
    orientation, colors = COUNTRY_FLAG_STRIPES.get(code, (_H, ["#ccc"]))
    n = len(colors)
    rects = []
    for i, color in enumerate(colors):
        if orientation == _H:
            rects.append(f'<rect y="{3 * i / n:.4f}" width="4" height="{3 / n:.4f}" fill="{color}"/>')
        else:
            rects.append(f'<rect x="{4 * i / n:.4f}" width="{4 / n:.4f}" height="3" fill="{color}"/>')
    extra = ""
    if code == "CZ":
        extra = '<polygon points="0,0 0,3 2,1.5" fill="#11457E"/>'
    svg = (
        '<svg class="flag-icon" viewBox="0 0 4 3" xmlns="http://www.w3.org/2000/svg" '
        'aria-hidden="true">' + "".join(rects) + extra + "</svg>"
    )
    return svg


def country_label(code: str) -> str:
    name = COUNTRY_NAMES.get(code, code)
    return format_html("{} {} ({})", mark_safe(country_flag_svg(code)), name, code)


COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "AT": (47.5, 14.5),
    "BE": (50.8, 4.5),
    "CZ": (49.8, 15.5),
    "DE": (51.2, 10.4),
    "FR": (46.6, 2.2),
    "HR": (45.1, 15.2),
    "HU": (47.2, 19.5),
    "LU": (49.8, 6.1),
    "NL": (52.1, 5.3),
    "PL": (51.9, 19.1),
    "RO": (45.9, 24.9),
    "SI": (46.1, 14.8),
    "SK": (48.7, 19.7),
}

DEFAULT_CENTER = (
    round(sum(lat for lat, _ in COUNTRY_CENTROIDS.values()) / len(COUNTRY_CENTROIDS), 2),
    round(sum(lon for _, lon in COUNTRY_CENTROIDS.values()) / len(COUNTRY_CENTROIDS), 2),
)
DEFAULT_ZOOM = 5
