"""
EC-Verspätungsprediktor — Streamlit-App
=======================================

Start:
    streamlit run app.py

Diese Datei enthält die komplette Weboberfläche für das bereits trainierte
Modell aus ``ec_models.pkl``. Die App lädt keine SBB-/OpenTransport-Ist-Daten
oder Fahrplandaten mehr von Bahn-Websites. Diese Abfrage war unzuverlässig und
wurde deshalb entfernt.

Was die App weiterhin live lädt:
    • Wetterdaten von Open-Meteo, weil Temperatur, Niederschlag, Schnee, Wind
      und Sichtweite Teil des trainierten Merkmalsvektors sind.
    • Aktuelle SBB-Störungsmeldungen als Kontextanzeige. Sie fliessen nicht in
      das Modell ein, weil im Training keine historischen Störungsmeldungen mit
      gleicher Struktur verfügbar sind.

Was lokal vorausgesetzt wird:
    • ``ec_models.pkl`` muss vorhanden sein. Diese Datei wird vom Trainingsskript
      ``ec_delay_predictor.py`` erzeugt.
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

matplotlib.use("Agg")  # Headless-Backend: Matplotlib braucht kein Display.

# ─── SEITENKONFIGURATION ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="EC-Verspätungsprediktor",
    page_icon="🚆",
    layout="centered",
)

# ─── KONSTANTEN ───────────────────────────────────────────────────────────────
#
# Diese Konstanten spiegeln die Trainingskonfiguration aus
# ec_delay_predictor.py. Sie werden hier absichtlich dupliziert, damit die
# Web-App auch ohne Import des Trainingsskripts lauffähig bleibt. Wichtig:
# Merkmalsnamen bleiben Englisch, weil sie exakt zu den im Modell gespeicherten
# Spaltennamen passen müssen.

STOPS_ORDERED = ["Zürich HB", "Zürich Flughafen", "Winterthur", "St. Gallen"]
STOP_TO_IDX = {s: i for i, s in enumerate(STOPS_ORDERED)}

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
    """
    Lädt die trainierten Modelle aus ``ec_models.pkl``.

    Streamlit cached das Ergebnis als Resource, weil das Modell über mehrere
    Interaktionen hinweg identisch bleibt und nicht bei jedem Klick neu von der
    Festplatte geladen werden muss.

    Rückgabe
    --------
    dict | None
        Modellpaket mit Klassifikator, Regressor und Merkmalsliste, oder None,
        wenn die Datei nicht vorhanden ist.
    """
    try:
        return joblib.load("ec_models.pkl")
    except FileNotFoundError:
        return None


# ─── WETTERDATEN ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather(stop: str, target_date: date) -> dict | None:
    """
    Lädt stündliche Wetterdaten für eine Haltestelle und einen Tag.

    Für vergangene Daten wird die Open-Meteo Archive API verwendet, für den
    heutigen und zukünftige Tage die Forecast API. Beide Schnittstellen liefern
    dieselben Variablennamen, sodass die restliche App nicht unterscheiden muss,
    woher die Daten stammen.

    Parameter
    ---------
    stop : str
        Name der Haltestelle, muss in ``STOP_COORDS`` vorhanden sein.
    target_date : date
        Tag, für den die 24 Stundenwerte geladen werden.

    Rückgabe
    --------
    dict | None
        Wörterbuch ``{stunde: wetterwerte}``. Bei Netzwerkfehlern oder leerer
        Antwort wird None zurückgegeben; die Merkmalserzeugung nutzt dann
        sichere Standardwerte.
    """
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
    """
    Gibt den Wetterdatensatz für eine Stunde zurück.

    Falls Open-Meteo keine exakt passende Stunde geliefert hat, wird die nächste
    verfügbare Stunde verwendet. Ohne Wetterdaten wird ein leeres Wörterbuch
    zurückgegeben; spätere Funktionen ersetzen fehlende Werte durch Defaults.
    """
    if not weather_by_hour:
        return {}
    if hour in weather_by_hour:
        return weather_by_hour[hour]
    nearest = min(weather_by_hour.keys(), key=lambda h: abs(h - hour))
    return weather_by_hour[nearest]


# ─── STÖRUNGEN AUS SBB OPEN DATA ─────────────────────────────────────────────

SBB_DISRUPTIONS_URL = (
    "https://data.sbb.ch/api/explore/v2.1/catalog/datasets"
    "/rail-traffic-information/records"
)

# Die SBB/Opendatasoft-Antworten können je nach API-Version leicht andere
# Feldnamen verwenden. Diese Kandidatenlisten halten die Auswertung robust.
_DESC_KEYS = ("description", "title", "beschreibung", "meldungstext")
_CAUSE_KEYS = ("cause", "ursache", "grund")
_STATUS_KEYS = ("status",)
_TYPE_KEYS = ("transporttype", "verkehrsart", "type")


def _pick(item: dict, keys: tuple[str, ...], default: str = "") -> str:
    """
    Gibt den ersten vorhandenen Wert aus einer Liste möglicher Feldnamen zurück.

    So kann dieselbe Logik mit deutschen und englischen SBB-Feldnamen umgehen,
    ohne an einer kleinen Schemaänderung der API zu scheitern.
    """
    for key in keys:
        value = item.get(key)
        if value:
            return str(value).strip()
    return default


@st.cache_data(ttl=300, show_spinner=False)
def fetch_disruptions(dep_date: str) -> list[dict]:
    """
    Lädt aktuelle Fernverkehrs-Störungen aus SBB Open Data.

    Die API liefert keine historischen Ist-Daten, sondern aktuelle bzw. aktive
    Verkehrsinformationen. Diese Störungen sind ausdrücklich weiterhin Teil der
    Website. ``dep_date`` wird genutzt, um Meldungen zu bevorzugen, die am
    gewählten Datum aktiv sind.

    Rückgabe
    --------
    list[dict]
        Normalisierte Meldungen mit ``type``, ``severity`` und ``description``.
        Bei API-Fehlern wird eine leere Liste zurückgegeben, damit die
        Vorhersage weiterhin funktioniert.
    """
    day_start = f"{dep_date}T00:00:00"
    day_end = f"{dep_date}T23:59:59"

    results = []
    for where_clause in [
        (
            "transporttype=\"Fernverkehr\""
            f" AND starttime<=date\"{day_end}\""
            f" AND (endtime>=date\"{day_start}\" OR endtime IS NULL)"
        ),
        "transporttype=\"Fernverkehr\"",
        None,
    ]:
        try:
            params: dict = {"limit": 100, "order_by": "starttime desc"}
            if where_clause:
                params["where"] = where_clause

            resp = requests.get(SBB_DISRUPTIONS_URL, params=params, timeout=12)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            break
        except Exception:
            results = []

    output = []
    for item in results:
        description = _pick(item, _DESC_KEYS)
        cause = _pick(item, _CAUSE_KEYS).lower()
        desc_lower = description.lower()
        status = _pick(item, _STATUS_KEYS).lower()

        transport = _pick(item, _TYPE_KEYS).lower()
        if transport and "fern" not in transport and "zug" not in transport \
                and "rail" not in transport and "train" not in transport:
            continue

        if any(k in cause or k in desc_lower
               for k in ("bau", "construction", "works",
                         "gleisarbeiten", "unterhaltsarbeiten", "baustelle")):
            dtype = "construction"
        elif any(k in cause or k in desc_lower
                 for k in ("signal", "strom", "power", "stellwerk", "weiche")):
            dtype = "signal_fault"
        elif any(k in cause or k in desc_lower
                 for k in ("wetter", "weather", "schnee", "snow", "wind", "sturm")):
            dtype = "weather_event"
        else:
            dtype = "other"

        if any(k in status or k in desc_lower
               for k in ("ausfall", "cancelled", "cancel", "totalausfall", "fällt aus")):
            severity = 3
        elif any(k in desc_lower for k in ("verspätung", "delay", "störung", "unterbruch")):
            severity = 2
        else:
            severity = 1

        output.append({
            "type": dtype,
            "severity": severity,
            "description": description or "(keine Beschreibung)",
        })

    return output


# ─── FEATURE-AUFBAU ───────────────────────────────────────────────────────────

def _wv(w: dict, key: str, default=np.nan):
    """
    Liest einen Wetterwert mit Fallback aus einem Wörterbuch.

    Open-Meteo kann einzelne Werte als None liefern. Für das Modell müssen aber
    alle Merkmalsspalten numerisch befüllt sein; deshalb ersetzt diese Funktion
    fehlende Einträge durch den übergebenen Standardwert.
    """
    v = w.get(key)
    return v if v is not None else default


def weather_record(w: dict, prefix: str) -> dict:
    """
    Wandelt Open-Meteo-Werte in Modellmerkmale um.

    ``prefix`` ist entweder ``orig`` für den Ursprungsbahnhof oder ``dest`` für
    den Zielbahnhof. So entstehen Spalten wie ``orig_temp`` und ``dest_temp``.
    """
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
    """
    Baut den vollständigen Merkmalsvektor für eine einzelne Vorhersage.

    Aus Route, Abfahrtszeit und Wetter entsteht genau eine DataFrame-Zeile. Die
    Spalten werden am Ende in dieselbe Reihenfolge gebracht, die beim Training
    gespeichert wurde. Fehlende Spalten werden mit 0 ergänzt, damit ältere
    Modellpakete mit den früheren Störungsmerkmalen weiterhin robust laden.
    """
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

    # Abgeleitete Merkmale: gleiche Berechnungen wie im Trainingsskript.
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

def route_diagram(origin: str, destination: str) -> str:
    """
    Erzeugt eine kompakte Markdown-Darstellung der gewählten Strecke.

    Der Start wird grün, das Ziel rot markiert. Haltestellen, die nicht auf dem
    aktuell gewählten Segment liegen, werden durchgestrichen. Die Funktion gibt
    nur Text zurück; Streamlit rendert ihn später mit ``st.markdown``.
    """
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

    arrow = " → " if forward else " ← "
    return arrow.join(parts)


def weather_icon(code: int | None) -> str:
    """Gibt ein Wettersymbol zum WMO-Wettercode zurück."""
    return WMO_ICONS.get(int(code) if code else 0, "🌡️")


def weather_description(code: int | None) -> str:
    """Gibt eine deutschsprachige Beschreibung zum WMO-Wettercode zurück."""
    return WMO_DESCRIPTIONS.get(int(code) if code else 0, "Unbekannt")


# ─── VISUALISIERUNGEN ────────────────────────────────────────────────────────

def _fig_to_img(fig) -> bytes:
    """
    Rendert eine Matplotlib-Figur als PNG-Bytefolge.

    Streamlit kann diese Bytes direkt mit ``st.image`` anzeigen. Nach dem
    Speichern wird die Figur geschlossen, damit bei wiederholten Interaktionen
    keine Matplotlib-Objekte im Speicher hängen bleiben.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_feature_importances(clf, feature_cols: list[str]) -> bytes:
    """
    Visualisiert die 15 wichtigsten Eingabemerkmale des Klassifikators.

    Die Werte stammen aus der Gini-Wichtigkeit des Random Forest. Hohe Balken
    bedeuten, dass das Merkmal häufig und wirkungsvoll für Baum-Splits genutzt
    wurde.
    """
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


def plot_delay_by_hour(clf, reg, feature_cols: list[str],
                       origin: str, destination: str) -> bytes:
    """
    Zeigt den modellierten Einfluss der Abfahrtsstunde.

    Die Funktion erzeugt künstliche Eingaben für alle Stunden von 0 bis 23 und
    hält Route, Wochentag und Wetter konstant. So wird sichtbar, wie sich die
    Vorhersage allein durch die Tageszeit verändert.
    """
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
        # Typische Schönwetterwerte als konstante Vergleichsbasis.
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

    # Beide Achsen haben eigene Linien; hier werden die Legenden zusammengeführt.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    fig.tight_layout()
    return _fig_to_img(fig)


def plot_delay_vs_weather(clf, reg, feature_cols: list[str],
                          origin: str, destination: str) -> bytes:
    """
    Zeigt den modellierten Einfluss des Schlechtwetter-Scores.

    Der Score wird von 0 bis 4 variiert. Aus jedem Score werden plausible
    Wetterwerte für Schnee, Regen, Windböen und Sichtweite abgeleitet.
    """
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
        # Score grob auf physische Werte zurückprojizieren:
        # score ≈ snow*1.5 + precip*1.0 + gusts*0.8 + visibility_loss*0.7
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

    # Dezente Bereiche helfen, milde, mittlere und starke Wetterlagen zu lesen.
    ax1.axvspan(0,   1.0, alpha=0.06, color="green",  label="Mild")
    ax1.axvspan(1.0, 2.5, alpha=0.06, color="orange", label="Mittel")
    ax1.axvspan(2.5, 4.0, alpha=0.06, color="red",    label="Stark")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    fig.tight_layout()
    return _fig_to_img(fig)


def plot_single_tree(clf, feature_cols: list[str]) -> bytes:
    """
    Visualisiert einen einzelnen Entscheidungsbaum aus dem Random Forest.

    Ein kompletter Baum wäre sehr gross. Deshalb werden nur die ersten drei
    Ebenen gezeigt, damit die wichtigsten frühen Entscheidungen lesbar bleiben.
    """
    estimator = clf.estimators_[0]
    fig, ax = plt.subplots(figsize=(18, 6))
    plot_tree(
        estimator,
        feature_names=feature_cols,
        class_names=["Pünktlich", "Verspätet"],
        filled=True,
        rounded=True,
        max_depth=3,          # Nur die oberen drei Ebenen zeigen.
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


# ─── HAUPT-APP ────────────────────────────────────────────────────────────────

def main():
    """
    Rendert die komplette Streamlit-Oberfläche und führt Vorhersagen aus.

    Ablauf:
      1. Modellpaket laden.
      2. Route, Datum und Abfahrtszeit aus der UI lesen.
      3. Wetter für Abfahrt und approximierte Ankunft laden.
      4. Klassifikator und Regressor ausführen.
      5. Aktuelle SBB-Störungen als Kontext laden und anzeigen.
      6. Ergebnis, Wetterwerte und Modellvisualisierungen anzeigen.
    """
    st.title("🚆 EC-Verspätungsprediktor")
    st.caption("Korridor Zürich HB ↔ St. Gallen · Modell + Open-Meteo-Wetterdaten")

    # ── Modelle laden ─────────────────────────────────────────────────────────
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
        dep_time_input = st.time_input("Abfahrtszeit", value=time(7, 7), step=60)

    dep_dt = datetime.combine(dep_date, dep_time_input)

    # ── Vorhersage-Button ─────────────────────────────────────────────────────
    st.divider()
    predict_clicked = st.button("🔮 Verspätung vorhersagen", type="primary", use_container_width=True)

    if not predict_clicked:
        st.info("Wählen Sie Route und Abfahrtszeit und klicken Sie dann auf **Verspätung vorhersagen**.")

        # Modellgrafiken bereits vor der ersten Vorhersage anzeigen.
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

    # ── SBB-Störungen laden ───────────────────────────────────────────────────
    with st.spinner("SBB-Störungen werden geprüft …"):
        disruptions = fetch_disruptions(dep_date.isoformat())

    # ── Ergebnis anzeigen ─────────────────────────────────────────────────────
    st.subheader("Vorhersage")

    if delay_class:
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
        f"<div style='border-radius:8px; padding:16px 20px; background:#1e1e2e;"
        f"font-size:1.25rem; font-weight:600'>"
        f"{verdict_color} &nbsp; {verdict_text}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    m1, m2, m3 = st.columns(3)
    m1.metric("Verspätungsrisiko", f"{delay_prob:.0%}")
    m2.metric("Erwartete Verspätung", f"{max(0, delay_mins):.1f} min")
    m3.metric("SBB-Schwelle", f"≥ {threshold} min = verspätet")

    # ── Wetterkarten ──────────────────────────────────────────────────────────
    st.subheader("Wetterbedingungen")
    w_col1, w_col2 = st.columns(2)

    def weather_card(col, stop: str, w: dict, label: str):
        """
        Rendert eine kompakte Wetterkarte für Abfahrt oder Ankunft.

        Die Karte zeigt nur die Modell-relevanten Hauptwerte. Fehlende Werte
        werden als Gedankenstrich angezeigt, damit die UI bei leeren
        Wetterantworten stabil bleibt.
        """
        wcode = int(w.get("weather_code") or 0)
        icon = weather_icon(wcode)
        desc = weather_description(wcode)
        with col:
            st.markdown(f"**{icon} {label} — {stop}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Temp.",
                      f"{w.get('temperature_2m', '—'):.1f} °C"
                      if w.get("temperature_2m") is not None else "—")
            c2.metric("Regen", f"{w.get('precipitation', 0):.1f} mm")
            c3.metric("Schnee", f"{w.get('snowfall', 0):.1f} cm")
            c4, c5, c6 = st.columns(3)
            c4.metric("Wind",
                      f"{w.get('wind_speed_10m', '—'):.0f} km/h"
                      if w.get("wind_speed_10m") is not None else "—")
            c5.metric("Böen",
                      f"{w.get('wind_gusts_10m', '—'):.0f} km/h"
                      if w.get("wind_gusts_10m") is not None else "—")
            c6.metric("Zustand", desc[:18])

    weather_card(w_col1, origin, w_orig, "Abfahrt")
    weather_card(w_col2, destination, w_dest, "Ankunft")

    # ── SBB-Störungsmeldungen ─────────────────────────────────────────────────
    if disruptions:
        st.subheader("⚠️ Aktive SBB-Störungen")
        severity_labels = ["Info", "Leicht", "Erheblich", "Ausfall"]
        type_labels = {
            "construction": "Baustelle / Gleisarbeiten",
            "signal_fault": "Signal- oder Stellwerksstörung",
            "weather_event": "Wetterbedingte Störung",
            "other": "Sonstige Störung",
        }
        for disruption in disruptions:
            severity = severity_labels[min(disruption["severity"], 3)]
            dtype = type_labels.get(disruption["type"], "Sonstige Störung")
            st.warning(
                f"**{dtype}** · {severity}\n\n"
                + (disruption.get("description") or "")
            )
    else:
        st.success("Keine aktiven Fernverkehrs-Störungen für dieses Datum gefunden.")

    # ── Modelleinblicke ───────────────────────────────────────────────────────
    _render_model_insights(clf, reg, feature_cols, origin, destination)


# ─── MODELLEINBLICKE ─────────────────────────────────────────────────────────

def _render_model_insights(clf, reg, feature_cols: list[str],
                           origin: str, destination: str) -> None:
    """
    Rendert die erklärenden Modellgrafiken.

    Die Expander zeigen:
      1. Wichtigste Merkmale des Klassifikators.
      2. Vorhersageverlauf nach Abfahrtsstunde.
      3. Vorhersageverlauf nach Wetterbelastung.
      4. Einen beispielhaften Entscheidungsbaum aus dem Random Forest.
    """
    st.divider()
    st.subheader("📊 Modelleinblicke")

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
