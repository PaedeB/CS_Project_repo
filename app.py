# Hinweis: Im Rahmen dieser Arbeit wurde KI-Unterstützung (Claude, Anthropic) eingesetzt —
# primär für Code-Review, Fehlersuche, Überarbeitung von Teilabschnitten sowie die
# Verbesserung von Lesbarkeit und Dokumentation. Konzeption, Modellwahl und inhaltliche
# Entscheidungen wurden eigenständig erarbeitet.

"""
EC-Verspätungsprediktor — Streamlit-App
Start: streamlit run app.py
Benötigt: ec_models.pkl, Internetzugang für Wetter- und Störungsdaten.
"""

import io
from datetime import date, datetime, time, timedelta

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.tree import plot_tree

matplotlib.use("Agg")

# ─── SEITENKONFIGURATION ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="EC-Verspätungsprediktor",
    page_icon="🚆",
    layout="centered",
)

# ─── KONSTANTEN ───────────────────────────────────────────────────────────────

STOPS_ORDERED = ["Zürich HB", "Zürich Flughafen", "Winterthur", "St. Gallen"]
STOP_TO_IDX = {s: i for i, s in enumerate(STOPS_ORDERED)}

# Koordinaten für den Wetter API - als Dictionary

STOP_COORDS = {
    "Zürich HB":         {"lat": 47.3779, "lon": 8.5403},
    "Zürich Flughafen": {"lat": 47.4504, "lon": 8.5624},
    "Winterthur":        {"lat": 47.4997, "lon": 8.7241},
    "St. Gallen":        {"lat": 47.4241, "lon": 9.3763},
}

WEATHER_VARIABLES = [
    "temperature_2m", "precipitation", "snowfall", "snow_depth",
    "wind_speed_10m", "wind_gusts_10m", "visibility",
    "cloud_cover", "relative_humidity_2m", "weather_code", "surface_pressure",
]

# Dicitionary für die Beschreibung des WMO-Codes

WMO_DESCRIPTIONS = {
    0: "Klar", 1: "Überwiegend klar", 2: "Teilweise bewölkt", 3: "Bedeckt",
    45: "Nebel", 48: "Raureifnebel",
    51: "Leichter Nieselregen", 53: "Mässiger Nieselregen", 55: "Starker Nieselregen",
    61: "Leichter Regen", 63: "Mässiger Regen", 65: "Starker Regen",
    71: "Leichter Schneefall", 73: "Mässiger Schneefall", 75: "Starker Schneefall",
    77: "Schneegriesel",
    80: "Leichte Schauer", 81: "Mässige Schauer", 82: "Heftige Schauer",
    85: "Leichte Schneeschauer", 86: "Starke Schneeschauer",
    95: "Gewitter", 96: "Gewitter mit Hagel", 99: "Gewitter mit starkem Hagel",
}

# Übersetzung der WMO-Wettercodes zu Symoblen für die Webseite

WMO_ICONS = {
    0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️",
    45: "🌫️", 48: "🌫️",
    51: "🌦️", 53: "🌦️", 55: "🌧️",
    61: "🌧️", 63: "🌧️", 65: "🌧️",
    71: "🌨️", 73: "🌨️", 75: "❄️", 77: "❄️",
    80: "🌦️", 81: "🌧️", 82: "⛈️",
    85: "🌨️", 86: "❄️",
    95: "⛈️", 96: "⛈️", 99: "⛈️",
}


# ─── MODELL LADEN ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Modelle werden geladen …")
def load_models():
    """Lädt trainierte Modelle aus ec_models.pkl."""
    try:
        return joblib.load("ec_models.pkl")
    except FileNotFoundError:
        return None


# ─── WETTERDATEN ──────────────────────────────────────────────────────────────

# Laden der Wetterdaten, gleich wie beim ec_delay_predictor.
# Wichtigste unterscheidung bei vergangenen Daten wird ein anderer API Link genommen
# somit handelt es sich um eine Unterscheidung zwischen den historischen Daten und
# den "Forecast"-Wetter.

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather(stop: str, target_date: date) -> dict | None:
    """Stündliche Wetterdaten für eine Haltestelle von Open-Meteo."""
    coords = STOP_COORDS[stop]
    today = date.today()
    date_str = target_date.isoformat()

    url = ("https://archive-api.open-meteo.com/v1/archive"
           if target_date <= today else
           "https://api.open-meteo.com/v1/forecast")

    try:
        resp = requests.get(url, params={
            "latitude":   coords["lat"],
            "longitude":  coords["lon"],
            "start_date": date_str,
            "end_date":   date_str,
            "hourly":     WEATHER_VARIABLES,
            "timezone":   "Europe/Zurich",
        }, timeout=20)
        resp.raise_for_status()
    except Exception as exc:
        st.warning(f"Wetterdaten für {stop} konnten nicht geladen werden: {exc}")
        return None

    hourly = resp.json().get("hourly", {})
    if not hourly.get("time"):
        return None

    result = {}
    times = hourly.pop("time", [])
    for i, ts in enumerate(times):
        hour = datetime.fromisoformat(ts).hour
        result[hour] = {k: (v[i] if v[i] is not None else 0) for k, v in hourly.items()}
    return result


def weather_for_hour(weather_by_hour: dict | None, hour: int) -> dict:
    """Wetterwerte für eine Stunde, mit Fallback auf nächste verfügbare."""
    if not weather_by_hour:
        return {}
    if hour in weather_by_hour:
        return weather_by_hour[hour]
    nearest = min(weather_by_hour.keys(), key=lambda h: abs(h - hour))
    return weather_by_hour[nearest]


# ─── EC-FAHRPLAN 2026 ────────────────────────────────────────────────────────
# Hardcodiert; Frühzug (Index 0) entfällt Sa + So.
# Hardcodierung für die Fahrtzeiten - für die Dropdownliste und die Plots

_EC_SCHEDULE: dict[str, dict[str, list[tuple[int,int]]]] = {
    "forward": {
        "Zürich HB":        [(5,33),(7,33),(9,33),(11,33),(13,33),(15,33),(17,33),(19,33)],
        "Zürich Flughafen": [(5,42),(7,42),(9,42),(11,42),(13,42),(15,42),(17,42),(19,42)],
        "Winterthur":       [(5,57),(7,57),(9,57),(11,57),(13,57),(15,57),(17,57),(19,57)],
    },
    "backward": {
        "St. Gallen":       [(6,29),(7,29),(9,29),(11,29),(13,29),(15,29),(17,29),(20,29)],
        "Winterthur":       [(7, 1),(8, 1),(10, 1),(12, 1),(14, 1),(16, 1),(18, 1),(21, 1)],
        "Zürich Flughafen": [(7,16),(8,16),(10,16),(12,16),(14,16),(16,16),(18,16),(21,16)],
    },
}

# Beschreibung der EC Fahrtnummern
_EC_TRAIN_NUMBERS: dict[str, list[str]] = {
    "forward":  ["EC 195","EC 197","EC 199","EC 191","EC 193","EC 195","EC 197","EC 199"],
    "backward": ["EC 194","EC 196","EC 198","EC 190","EC 192","EC 194","EC 196","EC 198"],
}


def fetch_connections(origin: str,
                      destination: str,
                      dep_date: date,
                      dep_time: time) -> list[dict]:
    """EC-Abfahrten am Ursprungsbahnhof aus dem hardcodierten Fahrplan 2026."""
    orig_idx = STOP_TO_IDX.get(origin, -1)
    dest_idx = STOP_TO_IDX.get(destination, -1)
    if orig_idx < 0 or dest_idx < 0 or orig_idx == dest_idx:
        return []

    direction  = "forward" if orig_idx < dest_idx else "backward"
    is_weekend = dep_date.weekday() >= 5
    times      = _EC_SCHEDULE[direction].get(origin, [])
    numbers    = _EC_TRAIN_NUMBERS[direction]

    if not times:
        return []

    results = []
    for i, (h, m) in enumerate(times):
        if is_weekend and i == 0:
            continue

        departure_raw = (
            f"{dep_date.strftime('%Y-%m-%d')}T{h:02d}:{m:02d}:00+01:00"
        )
        results.append({
            "train":     numbers[i % len(numbers)],
            "departure": departure_raw,
            "arrival":   None,
            "platform":  None,
            "duration":  None,
            "transfers": 0,
        })

    return results

# ─── FEATURE-AUFBAU ───────────────────────────────────────────────────────────

# Grundätzlich gleiche Logik zum Aufbauen der Features wie bei ec_delay_predictor.py

def _wv(w: dict, key: str, default=np.nan):
    """Wetterwert mit Fallback aus einem Wörterbuch lesen."""
    v = w.get(key)
    return v if v is not None else default


def weather_record(w: dict, prefix: str) -> dict:
    """Wetter-Dict in Merkmalseinträge mit Präfix umwandeln."""
    return {
        f"{prefix}_temp":         _wv(w, "temperature_2m"),
        f"{prefix}_precip":       _wv(w, "precipitation",     0),
        f"{prefix}_snow":         _wv(w, "snowfall",          0),
        f"{prefix}_snow_depth":   _wv(w, "snow_depth",        0),
        f"{prefix}_wind":         _wv(w, "wind_speed_10m"),
        f"{prefix}_gusts":        _wv(w, "wind_gusts_10m"),
        f"{prefix}_visibility":   _wv(w, "visibility",    10000),
        f"{prefix}_cloud":        _wv(w, "cloud_cover"),
        f"{prefix}_humidity":     _wv(w, "relative_humidity_2m"),
        f"{prefix}_pressure":     _wv(w, "surface_pressure"),
        f"{prefix}_weather_code": _wv(w, "weather_code",      0),
    }


def build_features(origin: str, destination: str, dep_dt: datetime,
                   w_orig: dict, w_dest: dict,
                   feature_cols: list[str]) -> pd.DataFrame:
    """Merkmalsvektor für eine Vorhersage erstellen."""
    orig_idx = STOP_TO_IDX[origin]
    dest_idx = STOP_TO_IDX[destination]
    direction = 0 if orig_idx < dest_idx else 1

    month        = dep_dt.month
    dep_hour     = dep_dt.hour
    day_of_week  = dep_dt.weekday()
    week_of_year = dep_dt.isocalendar().week

    record = {
        "direction":             direction,
        "origin_stop_idx":       orig_idx,
        "destination_stop_idx":  dest_idx,
        "segment_length":        abs(dest_idx - orig_idx),
        "dep_hour":     dep_hour,
        "day_of_week":  day_of_week,
        "month":        month,
        "week_of_year": week_of_year,
        "is_weekend":   int(day_of_week >= 5),
        "is_rush_hour": int(7 <= dep_hour <= 9 or 16 <= dep_hour <= 19),
        "season":       (month % 12) // 3,
        **weather_record(w_orig, "orig"),
        **weather_record(w_dest, "dest"),
    }

    record["bad_weather_score"] = (
          (min(record["orig_snow"],   5)     / 5)      * 1.5
        + (min(record["orig_precip"], 15)    / 15)     * 1.0
        + (min(record["orig_gusts"] or 0, 100) / 100)  * 0.8
        + ((10_000 - min(record["orig_visibility"] or 10000, 10000)) / 10_000) * 0.7
    )
    record["extreme_cold"]   = int((record["orig_temp"] or 0) < -5)
    record["extreme_heat"]   = int((record["orig_temp"] or 0) > 33)
    record["any_snow"]       = int((record["orig_snow"] or 0) > 0)
    record["heavy_rain"]     = int((record["orig_precip"] or 0) > 8)
    record["low_visibility"] = int((record["orig_visibility"] or 10000) < 1000)
    record["precip_diff"]    = abs((record["dest_precip"] or 0) - (record["orig_precip"] or 0))
    record["temp_diff"]      = abs((record["dest_temp"] or 0)   - (record["orig_temp"] or 0))
    record["wind_diff"]      = abs((record["dest_wind"] or 0)   - (record["orig_wind"] or 0))

    row = pd.DataFrame([record])
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    return row[feature_cols].fillna(0)


# ─── UI-HILFSFUNKTIONEN ──────────────────────────────────────────────────────

# Funktion zur visuellen Darstellung der Stops
def route_diagram(origin: str, destination: str) -> str:
    """Markdown-Darstellung der gewählten Strecke."""
    orig_idx = STOP_TO_IDX[origin]
    dest_idx = STOP_TO_IDX[destination]
    forward = orig_idx < dest_idx
    ordered = STOPS_ORDERED if forward else list(reversed(STOPS_ORDERED))
    min_i = min(orig_idx, dest_idx)
    max_i = max(orig_idx, dest_idx)
    on_route = {s for s, i in STOP_TO_IDX.items() if min_i <= i <= max_i}

    parts = []
    for stop in ordered:
        if stop == origin:
            parts.append(f"**🟢 {stop}**")
        elif stop == destination:
            parts.append(f"**🔴 {stop}**")
        elif stop in on_route:
            parts.append(f"⚪ {stop}")
        else:
            parts.append(f"〇 ~~{stop}~~")

    arrow = " → "
    return arrow.join(parts)

# Funktion zur Übersetzung des WMO-Codes zu einem Symbol
def weather_icon(code: int | None) -> str:
    """Wettersymbol zum WMO-Code."""
    return WMO_ICONS.get(int(code) if code else 0, "🌡️")

# Funktion zu Übersetzung des WMO-Codes zu der Beschreibung
def weather_description(code: int | None) -> str:
    """Deutschsprachige Beschreibung zum WMO-Code."""
    return WMO_DESCRIPTIONS.get(int(code) if code else 0, "Unbekannt")


# ─── VISUALISIERUNGEN ────────────────────────────────────────────────────────

def _fig_to_img(fig) -> bytes:
    """Matplotlib-Figur als PNG-Bytes rendern."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# Funktion zum Plotten der Wichtigkeiten der Features (Top 15)

def plot_feature_importances(clf, feature_cols: list[str]) -> bytes:
    """Top 15 Merkmalswichtigkeiten des Klassifikators."""
    imp = pd.Series(clf.feature_importances_, index=feature_cols).nlargest(15)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#e84040" if imp[f] > imp.median() else "#5b9cf6" for f in imp.index]
    ax.barh(imp.index[::-1], imp.values[::-1], color=colors[::-1])
    ax.set_xlabel("Wichtigkeit (Gini)", fontsize=9)
    ax.set_title("Top 15 Merkmalswichtigkeiten (Random-Forest-Klassifikator)", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return _fig_to_img(fig)

# Plot der Verspätungen pro Stunde 
# Zeigt bei konstantem Wetter und Datum die Veränderungen pro Uhrzeit
# Rot = Stündliche Versätungswahrscheinlichkeit
# Blau = Stündliche erwartete Verspätung (Regression)

def plot_delay_by_hour(clf, reg, feature_cols: list[str],
                       origin: str, destination: str) -> bytes:
    """Modellierter Einfluss der Abfahrtsstunde auf Verspätung."""
    base = {col: 0 for col in feature_cols}
    orig_idx = STOP_TO_IDX[origin]
    dest_idx = STOP_TO_IDX[destination]
    base.update({
        "direction":             0 if orig_idx < dest_idx else 1,
        "origin_stop_idx":       orig_idx,
        "destination_stop_idx":  dest_idx,
        "segment_length":        abs(dest_idx - orig_idx),
        "month":                 6,
        "day_of_week":           2,
        "week_of_year":          24,
        "is_weekend":            0,
        "season":                2,
        "orig_temp":         10.0,  "dest_temp":         10.0,
        "orig_wind":         15.0,  "dest_wind":         15.0,
        "orig_gusts":        25.0,  "dest_gusts":        25.0,
        "orig_visibility": 9000.0,  "dest_visibility": 9000.0,
        "orig_cloud":        40.0,  "dest_cloud":        40.0,
        "orig_humidity":     60.0,  "dest_humidity":     60.0,
        "orig_pressure":    1013.0, "dest_pressure":    1013.0,
        "bad_weather_score": 0.04,
    })

    hours = list(range(24))
    probs, mins_ = [], []
    for h in hours:
        row = {**base, "dep_hour": h,
               "is_rush_hour": int(7 <= h <= 9 or 16 <= h <= 19)}
        X = pd.DataFrame([row])[feature_cols].fillna(0)
        probs.append(clf.predict_proba(X)[0][1])
        mins_.append(max(0, float(reg.predict(X)[0])))

    fig, ax1 = plt.subplots(figsize=(8, 3.8))
    ax2 = ax1.twinx()

    ax1.plot(hours, [p * 100 for p in probs], color="#e84040",
             linewidth=2.2, marker="o", markersize=4, label="Verspätungswahrscheinlichkeit (%)")
    ax1.set_ylabel("Verspätungswahrscheinlichkeit (%)", color="#e84040", fontsize=9)
    ax1.tick_params(axis="y", labelcolor="#e84040", labelsize=8)
    ax1.set_ylim(0, 100)

    ax2.plot(hours, mins_, color="#5b9cf6",
             linewidth=2.2, marker="s", markersize=4, linestyle="--",
             label="Erwartete Verspätung (min)")
    ax2.set_ylabel("Erwartete Verspätung (min)", color="#5b9cf6", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#5b9cf6", labelsize=8)
    ax2.set_ylim(0, max(mins_) * 1.4 + 1)

    ax1.set_xlabel("Abfahrtsstunde", fontsize=9)
    ax1.set_xticks(hours)
    ax1.tick_params(axis="x", labelsize=7)
    ax1.set_title(
        f"Vorhersage nach Abfahrtsstunde\n"
        f"{origin} → {destination}  (Schönwetter, Wochentag)",
        fontsize=10,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.tight_layout()
    return _fig_to_img(fig)


# Der Plot stellt dar welche Verspätungswahrscheinlichkeit und Verspätungsdauer
# erwartet wird wenn sich der Wetterscore verändert

def plot_delay_vs_weather(clf, reg, feature_cols: list[str],
                          origin: str, destination: str) -> bytes:
    """Modellierter Einfluss des Schlechtwetter-Scores auf Verspätung."""
    base = {col: 0 for col in feature_cols}
    orig_idx = STOP_TO_IDX[origin]
    dest_idx = STOP_TO_IDX[destination]
    base.update({
        "direction":             0 if orig_idx < dest_idx else 1,
        "origin_stop_idx":       orig_idx,
        "destination_stop_idx":  dest_idx,
        "segment_length":        abs(dest_idx - orig_idx),
        "dep_hour":              8,
        "month":                 1,
        "day_of_week":           1,
        "week_of_year":          3,
        "is_weekend":            0,
        "is_rush_hour":          1,
        "season":                3,
        "orig_temp":             2.0, "dest_temp":             2.0,
        "orig_visibility":    5000.0, "dest_visibility":    5000.0,
        "orig_cloud":           80.0, "dest_cloud":           80.0,
        "orig_humidity":        85.0, "dest_humidity":        85.0,
        "orig_pressure":      1005.0, "dest_pressure":      1005.0,
    })

    scores = np.linspace(0, 4, 50)
    probs, mins_ = [], []
    for s in scores:
        snow = min(s / 1.5, 5.0)
        precip = min(s / 1.0, 15.0)
        gusts = min(s / 0.8, 100.0)
        vis = max(10000 - s * 0.7 * 10000, 100)
        row = {
            **base,
            "bad_weather_score": s,
            "orig_snow":          snow,  "dest_snow":          snow,
            "orig_precip":        precip, "dest_precip":       precip,
            "orig_gusts":         gusts, "dest_gusts":         gusts,
            "orig_wind":          gusts * 0.7, "dest_wind":    gusts * 0.7,
            "orig_visibility":    vis,   "dest_visibility":   vis,
            "any_snow":           int(snow > 0),
            "heavy_rain":         int(precip > 8),
            "low_visibility":     int(vis < 1000),
            "precip_diff":        0.0,
            "temp_diff":          0.0,
            "wind_diff":          0.0,
        }
        X = pd.DataFrame([row])[feature_cols].fillna(0)
        probs.append(clf.predict_proba(X)[0][1])
        mins_.append(max(0, float(reg.predict(X)[0])))

    fig, ax1 = plt.subplots(figsize=(8, 3.8))
    ax2 = ax1.twinx()

    ax1.plot(scores, [p * 100 for p in probs], color="#e84040",
             linewidth=2.2, label="Verspätungswahrscheinlichkeit (%)")
    ax1.set_ylabel("Verspätungswahrscheinlichkeit (%)", color="#e84040", fontsize=9)
    ax1.tick_params(axis="y", labelcolor="#e84040", labelsize=8)
    ax1.set_ylim(0, 100)

    ax2.plot(scores, mins_, color="#5b9cf6",
             linewidth=2.2, linestyle="--", label="Erwartete Verspätung (min)")
    ax2.set_ylabel("Erwartete Verspätung (min)", color="#5b9cf6", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#5b9cf6", labelsize=8)
    ax2.set_ylim(0, max(mins_) * 1.4 + 1)

    ax1.set_xlabel("Schlechtwetter-Score  (0 = ruhig · 4 = Schnee und Sturm)", fontsize=9)
    ax1.set_title(
        f"Vorhersage nach Wetterbelastung\n"
        f"{origin} → {destination}  (Morgenverkehr, Januar)",
        fontsize=10,
    )

    ax1.axvspan(0,   1.0, alpha=0.06, color="green",  label="Mild")
    ax1.axvspan(1.0, 2.5, alpha=0.06, color="orange", label="Mittel")
    ax1.axvspan(2.5, 4.0, alpha=0.06, color="red",    label="Stark")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    fig.tight_layout()
    return _fig_to_img(fig)


# Zeigt den Random Forrest grafisch (Top 3 Stufen)

def plot_single_tree(clf, feature_cols: list[str]) -> bytes:
    """Einen Entscheidungsbaum aus dem Random Forest visualisieren (3 Ebenen)."""
    estimator = clf.estimators_[0]
    fig, ax = plt.subplots(figsize=(18, 6))
    plot_tree(
        estimator,
        feature_names=feature_cols,
        class_names=["Pünktlich", "Verspätet"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=7,
        ax=ax,
        impurity=False,
        proportion=True,
    )
    ax.set_title(
        "Ein Entscheidungsbaum aus dem Random Forest (erste 3 Ebenen)\n"
        "Jeder Knoten zeigt: Split-Bedingung · Klassenanteile · Mehrheitsklasse",
        fontsize=10,
    )
    fig.tight_layout()
    return _fig_to_img(fig)

# ─── REGRESSIONSPLOT ─────────────────────────────────────────────────────────

# Regressionsplot um den User zu helfen eine Entscheidung zu treffen, wann
# zeitlich die beste Zeit den Zug zu nehmen ist (be kontstantem Wetter und Datum)

def plot_regression_by_hour(reg, feature_cols: list[str],
                            origin: str, destination: str,
                            target_date: date,
                            orig_weather_all: dict | None,
                            dest_weather_all: dict | None) -> bytes:
    """Regressorvorhersage für die tatsächlichen EC-Abfahrtszeiten mit echtem Tageswetter."""
    orig_idx = STOP_TO_IDX[origin]
    dest_idx = STOP_TO_IDX[destination]

    base = {col: 0 for col in feature_cols}
    base.update({
        "direction":             0 if orig_idx < dest_idx else 1,
        "origin_stop_idx":       orig_idx,
        "destination_stop_idx":  dest_idx,
        "segment_length":        abs(dest_idx - orig_idx),
        "month":                 target_date.month,
        "day_of_week":           target_date.weekday(),
        "week_of_year":          target_date.isocalendar().week,
        "is_weekend":            int(target_date.weekday() >= 5),
        "season":                (target_date.month % 12) // 3,
    })

    direction_key = "forward" if orig_idx < dest_idx else "backward"
    schedule_times = _EC_SCHEDULE[direction_key].get(origin, [])
    if not schedule_times:
        schedule_times = [(h, 0) for h in range(24)]

    delays = []
    x_labels = []

    for h, m in schedule_times:
        w_orig = weather_for_hour(orig_weather_all, h)
        w_dest = weather_for_hour(dest_weather_all, h)

        row = {
            **base,
            "dep_hour":     h,
            "is_rush_hour": int(7 <= h <= 9 or 16 <= h <= 19),
            **weather_record(w_orig, "orig"),
            **weather_record(w_dest, "dest"),
        }

        orig_snow       = row.get("orig_snow") or 0
        orig_precip     = row.get("orig_precip") or 0
        orig_gusts      = row.get("orig_gusts") or 0
        orig_visibility = row.get("orig_visibility") or 10000
        orig_temp       = row.get("orig_temp") or 0

        row["bad_weather_score"] = (
              (min(orig_snow, 5)       / 5)       * 1.5
            + (min(orig_precip, 15)    / 15)      * 1.0
            + (min(orig_gusts, 100)    / 100)     * 0.8
            + ((10_000 - min(orig_visibility, 10_000)) / 10_000) * 0.7
        )
        row["extreme_cold"]   = int(orig_temp < -5)
        row["extreme_heat"]   = int(orig_temp > 33)
        row["any_snow"]       = int(orig_snow > 0)
        row["heavy_rain"]     = int(orig_precip > 8)
        row["low_visibility"] = int(orig_visibility < 1000)
        row["precip_diff"]    = abs((row.get("dest_precip") or 0) - orig_precip)
        row["temp_diff"]      = abs((row.get("dest_temp") or 0)   - orig_temp)
        row["wind_diff"]      = abs((row.get("dest_wind") or 0)   - (row.get("orig_wind") or 0))

        X = pd.DataFrame([row])[feature_cols].fillna(0)
        pred = max(0, float(reg.predict(X)[0]))
        delays.append(pred)
        x_labels.append(f"{h:02d}:{m:02d}")

    x_pos = list(range(len(schedule_times)))

    fig, ax = plt.subplots(figsize=(8, 3.8))

    ax.plot(x_pos, delays, linewidth=2.5, marker="o", markersize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, fontsize=7)
    ax.set_ylabel("Erwartete Verspätung (min)", fontsize=9)
    ax.set_xlabel("Abfahrtszeit", fontsize=9)
    ax.set_title(
        f"Erwartete Verspätung nach Abfahrtszeit — {target_date.strftime('%d.%m.%Y')}\n"
        f"{origin} → {destination}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return _fig_to_img(fig)


# ─── HAUPT-APP ────────────────────────────────────────────────────────────────

def main():
    """Streamlit-Oberfläche rendern und Vorhersagen ausführen."""
    st.title("🚆 EC-Verspätungsprädiktor")
    st.caption("Korridor Zürich HB ↔ St. Gallen · Modell + Open-Meteo-Wetterdaten")

    models = load_models()
    if models is None:
        st.error("**Kein trainiertes Modell gefunden.**")
        st.info(
            "Trainieren Sie zuerst das Modell:\n\n"
            "```bash\n"
            "python ec_delay_predictor.py --start 2026-01-01 --end 2026-01-31\n"
            "```\n\n"
            "Die benötigten Ist-Daten müssen lokal im Ordner `istdaten_cache/` liegen."
        )
        return

    clf = models["clf"]
    reg = models["reg"]
    feature_cols = models["feature_cols"]
    threshold = models.get("threshold", 3)

    # ── Routenauswahl ─────────────────────────────────────────────────────────
    st.subheader("Route")
    col_from, col_arrow, col_to = st.columns([5, 1, 5])

    with col_from:
        origin = st.selectbox("Von", STOPS_ORDERED, index=0, key="origin")

    with col_arrow:
        st.markdown(
            "<div style='padding-top:1.9rem; text-align:center; font-size:1.4rem'>→</div>",
            unsafe_allow_html=True,
        )

    with col_to:
        dest_options = [s for s in STOPS_ORDERED if s != origin]
        default_dest = "St. Gallen" if origin != "St. Gallen" else "Zürich HB"
        dest_idx_in_opts = dest_options.index(default_dest) if default_dest in dest_options else 0
        destination = st.selectbox("Nach", dest_options,
                                   index=dest_idx_in_opts, key="destination")

    orig_idx = STOP_TO_IDX[origin]
    dest_idx = STOP_TO_IDX[destination]
    direction_lbl = "Zürich → St. Gallen" if orig_idx < dest_idx else "St. Gallen → Zürich"
    st.caption(f"Richtung: {direction_lbl} · Segment: {abs(dest_idx - orig_idx)} Halt(e)")
    st.markdown(route_diagram(origin, destination))

    # ── Datum und Uhrzeit ─────────────────────────────────────────────────────
    st.subheader("Abfahrt")
    col_date, col_time = st.columns(2)

    with col_date:
        dep_date = st.date_input(
            "Datum",
            value=date.today(),
            min_value=date.today() - timedelta(days=365),
            max_value=date.today() + timedelta(days=14),
        )

    with col_time:
        with st.spinner("EC-Züge werden geladen …"):
            ec_connections = fetch_connections(
                origin, destination, dep_date, time(5, 0)
            )

        if ec_connections:
            def _parse_dep(conn: dict) -> time | None:
                raw = conn.get("departure")
                if not raw:
                    return None
                try:
                    return datetime.fromisoformat(raw).time().replace(second=0, microsecond=0)
                except ValueError:
                    return None

            seen = set()
            options = []
            for conn in ec_connections:
                t = _parse_dep(conn)
                if t is not None and t not in seen:
                    seen.add(t)
                    options.append((t, conn))

            if options:
                default_idx = next(
                    (i for i, (t, _) in enumerate(options) if t >= time(7, 0)),
                    0,
                )

                def _label(item: tuple) -> str:
                    t, conn = item
                    train = conn.get("train", "EC")
                    platform = conn.get("platform")
                    plat_str = f" · Gleis {platform}" if platform else ""
                    return f"{t.strftime('%H:%M')} Uhr  ({train}{plat_str})"

                selected_item = st.selectbox(
                    "EC-Abfahrt",
                    options=options,
                    index=default_idx,
                    format_func=_label,
                    key="ec_dep_time",
                    help="Nur direkte EC-Verbindungen auf dieser Strecke.",
                )
                dep_time_input = selected_item[0]
            else:
                st.warning("Keine EC-Abfahrten gefunden – bitte Route oder Datum prüfen.")
                st.stop()
        else:
            st.warning("Fahrplandaten konnten nicht geladen werden.")
            st.stop()

    dep_dt = datetime.combine(dep_date, dep_time_input)

    # ── Vorhersage-Button ─────────────────────────────────────────────────────
    st.divider()
    predict_clicked = st.button("🔮 Verspätung vorhersagen", type="primary", use_container_width=True)

    if not predict_clicked:
        st.info("Wählen Sie Route und Abfahrtszeit und klicken Sie dann auf **Verspätung vorhersagen**.")
        _render_model_insights(clf, reg, feature_cols, origin, destination)
        return

    # ── Wetter laden ──────────────────────────────────────────────────────────
    arr_dt = dep_dt + timedelta(hours=1, minutes=6)

    with st.spinner("Wetterdaten werden geladen …"):
        orig_weather_all = fetch_weather(origin, dep_date)
        dest_weather_all = fetch_weather(destination, arr_dt.date())

    w_orig = weather_for_hour(orig_weather_all, dep_dt.hour)
    w_dest = weather_for_hour(dest_weather_all, arr_dt.hour)

    # ── Vorhersage berechnen ──────────────────────────────────────────────────
    X = build_features(origin, destination, dep_dt, w_orig, w_dest, feature_cols)

    delay_prob = clf.predict_proba(X)[0][1]
    delay_class = clf.predict(X)[0]
    delay_mins = float(reg.predict(X)[0])

   
    # ── Ergebnis anzeigen ─────────────────────────────────────────────────────
    st.subheader("Vorhersage")

    if delay_mins >= 3:
        if delay_mins >= 10:
            verdict_color = "🔴"
            verdict_text = "Deutliche Verspätung erwartet"
        else:
            verdict_color = "🟠"
            verdict_text = "Leichte Verspätung wahrscheinlich"
    else:
        verdict_color = "🟢"
        verdict_text = "Voraussichtlich pünktlich"

    st.markdown(
        f"<div style='border-radius:8px; padding:16px 20px; background:#f0f2f6;"
        f"font-size:1.25rem; font-weight:600'>"
        f"{verdict_color} &nbsp; {verdict_text}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.caption(f"SBB-Schwelle: ≥{threshold} min = verspätet, >= 10 min deutliche Verspätung")

    m1, m2 = st.columns(2)
    m1.metric("Verspätungsrisiko", f"{delay_prob:.0%}")
    m2.metric("Erwartete Verspätung", f"{max(0, delay_mins):.1f} min")

    # ── Wetterkarten ──────────────────────────────────────────────────────────
    st.subheader("Wetterbedingungen")

    def weather_card(w: dict):
        """Kompakte Wetterkarte mit Hauptwerten."""
        wcode = int(w.get("weather_code") or 0)
        desc = weather_description(wcode)

        c1, c2 = st.columns(2)
        c1.metric("Temp.",
        f"{w.get('temperature_2m', '—'):.1f} °C"
            if w.get("temperature_2m") is not None else "—")
        c2.metric("Regen", f"{w.get('precipitation', 0):.1f} mm")

        c3, c4 = st.columns(2)
        c3.metric("Schnee", f"{w.get('snowfall', 0):.1f} cm")
        c4.metric("Wind",
            f"{w.get('wind_speed_10m', '—'):.0f} km/h"
            if w.get("wind_speed_10m") is not None else "—")

        c5, c6 = st.columns(2)
        c5.metric("Böen",
            f"{w.get('wind_gusts_10m', '—'):.0f} km/h"
            if w.get("wind_gusts_10m") is not None else "—")
        c6.metric("Zustand", desc[:18])

    icon_dest = weather_icon(int(w_dest.get("weather_code") or 0))
    icon_orig = weather_icon(int(w_orig.get("weather_code") or 0))

    with st.expander((f"**{icon_orig} Abfahrt — {origin}**"), expanded=False):
        weather_card(w_orig)

    with st.expander((f"**{icon_dest} Ankunft — {destination}**"), expanded=False):
        weather_card(w_dest)

    st.subheader("Übersicht für die verschiedenen Abfahrtszeiten")

    with st.expander("**Bei welchen Zeiten sind welche Verspätungen erwartet?**", expanded=False):
        st.caption(
            f"Vorhersage des Regressors für alle EC-Abfahrten am {dep_date.strftime('%d.%m.%Y')} "
            "mit dem tatsächlichen Tageswetter. So siehst du, welcher Zug heute am wenigsten Verspätung hat."
        )
        st.image(
            plot_regression_by_hour(reg, feature_cols, origin, destination,
                                    dep_date, orig_weather_all, dest_weather_all),
            use_container_width=True,
        )
    _render_model_insights(clf, reg, feature_cols, origin, destination)


# ─── MODELLEINBLICKE ─────────────────────────────────────────────────────────

def _render_model_insights(clf, reg, feature_cols: list[str],
                           origin: str, destination: str) -> None:
    """Erklärende Modellgrafiken in Expandern anzeigen."""
    st.divider()
    st.subheader("📊 Modelleinblicke (für Neugierige)")

    with st.expander("Merkmalswichtigkeiten", expanded=False):
        st.caption(
            "Diese Balken zeigen, welche Eingaben der Klassifikator im Mittel "
            "am stärksten nutzt."
        )
        st.image(plot_feature_importances(clf, feature_cols), use_container_width=True)

    with st.expander("Verspätung nach Abfahrtsstunde", expanded=False):
        st.caption(
            "Rot zeigt die Verspätungswahrscheinlichkeit, blau die erwarteten "
            "Minuten. Route und Wetter bleiben in dieser Analyse konstant."
        )
        st.image(
            plot_delay_by_hour(clf, reg, feature_cols, origin, destination),
            use_container_width=True,
        )

    with st.expander("Verspätung nach Wetterbelastung", expanded=False):
        st.caption(
            "Der Schlechtwetter-Score wird von ruhigen Bedingungen bis zu "
            "starker Winterbelastung variiert."
        )
        st.image(
            plot_delay_vs_weather(clf, reg, feature_cols, origin, destination),
            use_container_width=True,
        )

    with st.expander("Entscheidungsbaum (erste 3 Ebenen)", expanded=False):
        st.caption(
            "Ein Baum aus dem Random Forest, auf drei Ebenen gekürzt. Blau "
            "steht für pünktlich, Orange für verspätet."
        )
        st.image(plot_single_tree(clf, feature_cols), use_container_width=True)



if __name__ == "__main__":
    main()