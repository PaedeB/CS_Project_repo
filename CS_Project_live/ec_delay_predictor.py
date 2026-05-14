# ─── SETUP & AUSFÜHRUNG ──────────────────────────────────────────────────────
#
# Schritt 1 — Ist-Daten herunterladen:
#   https://opentransportdata.swiss/de/dataset/istdaten
#   → Monat auswählen (z.B. Januar 2026), ZIP herunterladen.
#
# Schritt 2 — ZIP entpacken:
#   ZIP öffnen und die enthaltene CSV-Datei (z.B. 2026-01-01_istdaten.csv)
#   in den Ordner  CS_Project_live/istdaten_cache/  kopieren.
#   (Ordner manuell erstellen falls nicht vorhanden)
#
# Schritt 3 — Skript ausführen (im Ordner CS_Project_live/):
#   python ec_delay_predictor.py --start 2026-01-01 --end 2026-01-31
#
# Ausgabe: ec_models.pkl  (Modell für die Streamlit-App)
#          ec_delay_dataset.csv  (aufbereitete Trainingsdaten)
#
# ─────────────────────────────────────────────────────────────────────────────

# Hinweis: Im Rahmen dieser Arbeit wurde KI-Unterstützung (Claude, Anthropic 4.6 [01.08.2025]) eingesetzt —
# primär für Code-Review, Fehlersuche, Überarbeitung von Teilabschnitten sowie die
# Verbesserung von Lesbarkeit und Dokumentation. Konzeption, Modellwahl und inhaltliche
# Entscheidungen wurden eigenständig erarbeitet.


"""
EC-Verspätungsprediktor — Trainingsskript

Trainiert zwei Random-Forest-Modelle auf SBB Ist-Daten für die Strecke
Zürich HB ↔ St. Gallen:
  1. Klassifikator: Wird der Zug ≥ 3 min verspätet? (Ja/Nein)
  2. Regressor:     Wie viele Minuten Verspätung sind zu erwarten?

Verwendung:
    python ec_delay_predictor.py --start 2024-01-01 --end 2024-12-31


    
Ist-Daten: https://opentransportdata.swiss/de/dataset/istdaten
           → CSV/GZ-Dateien in istdaten_cache/ ablegen.
Ausgabe:   ec_models.pkl, ec_delay_dataset.csv
Modellarchitektur
-----------------
Es werden zwei Modelle trainiert:

1. Klassifikator (RandomForestClassifier):
   Frage: "Wird der Zug ≥ 3 Minuten verspätet ankommen?" (Ja/Nein)

2. Regressor (RandomForestRegressor):
   Frage: "Wie viele Minuten Verspätung sind zu erwarten?"

Beide Modelle verwenden dieselben 42 Merkmale aus den Kategorien:
  • Streckeninformation (Richtung, Haltestellen-Indizes, Segmentlänge)
  • Zeitmerkmale (Stunde, Wochentag, Monat, Saison, …)
  • Wetterdaten am Abfahrts- und Ankunftsort
  • Abgeleitete Merkmale (Schlechtwetter-Score, Schneefall-Flag, …)

Warum sind SBB-Störungen keine Modellmerkmale?
----------------------------------------------
Historische Störungen sind über die verwendeten Datenquellen nicht zuverlässig
abrufbar. Da die historischen Verspätungen aus den Ist-Daten den Einfluss
vergangener Störungen bereits indirekt enthalten, würden konstant leere
Störungsspalten dem Modell nichts beibringen.

Abgedeckte Haltestellen (beide Richtungen)
------------------------------------------
  Zürich HB → Zürich Flughafen → Winterthur → St. Gallen  (und umgekehrt)
"""

import argparse
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Scikit-learn gibt bei unbalancierten Klassen Warnungen aus — hier unterdrückt.
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Der EC hält bei den Haltestellen Zürich HB, Zürich Flughaven, Winterthur und St. Gallen
STOPS_ORDERED = ["Zürich HB", "Zürich Flughafen", "Winterthur", "St. Gallen"]
STOP_TO_IDX = {s: i for i, s in enumerate(STOPS_ORDERED)}

STOP_COORDS = {
    "Zürich HB":         {"lat": 47.3779, "lon": 8.5403},
    "Zürich Flughafen": {"lat": 47.4504, "lon": 8.5624},
    "Winterthur":        {"lat": 47.4997, "lon": 8.7241},
    "St. Gallen":        {"lat": 47.4241, "lon": 9.3763},
}

# Open-Meteo Archive API — stündliche historische Wetterdaten, kostenlos
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
ISTDATEN_CACHE_DIR = Path("istdaten_cache")
MODEL_PATH = Path("ec_models.pkl")

# SBB-Definition: Zug gilt als verspätet ab ≥ 3 Minuten Ankunftsverspätung
DELAY_THRESHOLD_MIN = 3

WEATHER_VARIABLES = [
    "temperature_2m",        # Lufttemperatur in 2 m Höhe [°C]
    "precipitation",         # Niederschlag [mm]
    "snowfall",              # Schneefall [cm]
    "snow_depth",            # Schneehöhe am Boden [m]
    "wind_speed_10m",        # Windgeschwindigkeit in 10 m Höhe [km/h]
    "wind_gusts_10m",        # Windböen in 10 m Höhe [km/h]
    "visibility",            # Sichtweite [m]
    "cloud_cover",           # Bewölkungsgrad [%]
    "relative_humidity_2m",  # Relative Luftfeuchtigkeit [%]
    "weather_code",          # WMO-Wettercode (kodierter Zustand wie Regen, Schnee …)
    "surface_pressure",      # Luftdruck auf Bodenniveau [hPa]
]

# Spaltennamen aus den Ist-Daten-CSV-Dateien, die wir benötigen.
# Alle anderen Spalten werden beim Einlesen ignoriert (spart RAM).
_IST_COLS = [
    "BETRIEBSTAG",           # Betriebsdatum (DD.MM.YYYY oder YYYY-MM-DD)
    "FAHRT_BEZEICHNER",      # Eindeutige Fahrt-ID — identifiziert eine einzelne Zugsfahrt
    "BETREIBER_ABK",         # Kürzel des Transportunternehmens (z. B. SBB)
    "PRODUKT_ID",            # Produkttyp — in Ist-Daten "Zug" oder "Bus" (NICHT "EC"!)
    "LINIEN_ID",             # Numerische Linien-ID
    "LINIEN_TEXT",           # Lesbarer Linienname, z. B. "EC", "IC", "S1"
    "ZUSATZFAHRT_TF",        # Flag: Zusatzfahrt (1 = ja)
    "FAELLT_AUS_TF",         # Flag: Fahrt ausgefallen (1 = ja)
    "BPUIC",                 # Numerischer Haltestellen-Code (UIC)
    "HALTESTELLEN_NAME",     # Name der Haltestelle (Text), z. B. "Zürich HB"
    "ANKUNFTSZEIT",          # Geplante Ankunftszeit
    "AN_PROGNOSE",           # Tatsächliche/prognostizierte Ankunftszeit
    "AN_PROGNOSE_STATUS",    # Status der Ankunftsprognose: REAL, PROGNOSE oder UNBEKANNT
    "ABFAHRTSZEIT",          # Geplante Abfahrtszeit
    "AB_PROGNOSE",           # Tatsächliche/prognostizierte Abfahrtszeit
    "AB_PROGNOSE_STATUS",    # Status der Abfahrtsprognose
    "DURCHFAHRT_TF",         # Flag: Halt ohne Türöffnung (Durchfahrt, 1 = ja)
]


# ══════════════════════════════════════════════════════════════════════════════
# ABSCHNITT 1 — IST-DATEN: LESEN AUS LOKALEM CACHE
# ══════════════════════════════════════════════════════════════════════════════
#
# Wichtig: PRODUKT_ID ist in diesen Dateien "Zug" — NICHT "EC".
# EC-Züge werden ausschliesslich über LINIEN_TEXT = "EC" identifiziert.

def _open_csv(path: Path):
    """Öffnet eine Ist-Daten-Datei direkt (.csv)"""
    return open(path, encoding="utf-8-sig")


def _normalize_date(date_str: str) -> str:
    """
    Normalisiert ein Datum in das ISO-Format YYYY-MM-DD.
    Unterstützt: DD.MM.YYYY, DD/MM/YYYY, YYYYMMDD und ISO-Format.
    Unbekannte Formate werden unverändert zurückgegeben.
    """
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Unbekanntes Format: unverändert zurückgeben
    return date_str


def _parse_ist_time(date_str: str, cell) -> datetime | None:
    """
    Parst eine Zeitzelle aus den Ist-Daten in ein datetime-Objekt.
    Variante 1: vollständiger Zeitstempel ("15.03.2024 08:32:00")
    Variante 2: nur Uhrzeit ("08:32:00") — wird mit Betriebsdatum kombiniert.
    """
    if not cell or pd.isna(cell):
        return None
    ts = str(cell).strip()

    # Variante 1: Vollständiger Zeitstempel
    for fmt in (
        "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%d.%m.%Y %H:%M",
    ):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            pass

    # Variante 2: Nur Zeit — mit normalisiertem Betriebsdatum kombinieren
    norm = _normalize_date(date_str)
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(ts, fmt).time()
            return datetime.combine(datetime.strptime(norm, "%Y-%m-%d").date(), t)
        except ValueError:
            pass

    return None  # Kein Format erkannt


def parse_istdaten_file(path: Path) -> pd.DataFrame:
    """
    Liest eine Ist-Daten-CSV und extrahiert alle EC-Halte auf der Strecke.
    Filter: LINIEN_TEXT beginnt mit "EC" (nicht PRODUKT_ID, da SBB dort
    "Zug" statt "EC" einträgt). Trennzeichen wird automatisch erkannt.
    """
    # ── Trennzeichen automatisch erkennen ─────────────────────────────────────
    try:
        with _open_csv(path) as fh:
            sample = fh.read(4096)
        sep = ";" if sample.count(";") > sample.count(",") else ","

        with _open_csv(path) as fh:
            df = pd.read_csv(
                fh,
                sep=sep,
                usecols=lambda c: c.upper().strip() in _IST_COLS,
                dtype=str,
                low_memory=False,
            )
    except Exception as exc:
        print(f"    ⚠️  {path.name}: Fehler beim Lesen — {exc}")
        return pd.DataFrame()
    #Ergänzung durch Chatgpt: Direktes einlesen der CSV - hatte einige Probleme

    # Spaltennamen vereinheitlichen (Grossschreibung, keine Leerzeichen)
    df.columns = [c.upper().strip() for c in df.columns]

    # ── Auf EC-Züge filtern ────────────────────────────────────────────────────
    ec_mask = df["LINIEN_TEXT"].str.strip().str.upper().str.startswith("EC", na=False)
    df = df[ec_mask]
    
    # ── Auf Haltestellen der Strecke filtern ──────────────────────────────────
    if "HALTESTELLEN_NAME" in df.columns:
        df = df[df["HALTESTELLEN_NAME"].isin(set(STOPS_ORDERED))]

    # ── Zeilen in strukturierte Datensätze umwandeln ──────────────────────────
    rows = []
    for _, row in df.iterrows():
        raw_date = str(row.get("BETRIEBSTAG", "")).strip()
        date_str = _normalize_date(raw_date)   # Immer YYYY-MM-DD ab hier
        stop     = str(row.get("HALTESTELLEN_NAME", "")).strip()
        linien   = str(row.get("LINIEN_TEXT", "")).strip()
        fahrt_id = str(row.get("FAHRT_BEZEICHNER", "")).strip()
    #dieser Teil wurde mit KI untestützt erstellt

        # FAELLT_AUS_TF: "1" oder "true" bedeutet die Fahrt ist ausgefallen
        cancelled = str(row.get("FAELLT_AUS_TF", "0")).strip() in ("1", "true", "True")

        rows.append({
            "date":          date_str,
            "fahrt_id":      fahrt_id,   # Eindeutige Fahrt-ID für spätere Gruppierung
            "linien_text":   linien,
            "stop":          stop,
            # Abfahrt: geplant und tatsächlich
            "dep_scheduled": _parse_ist_time(date_str, row.get("ABFAHRTSZEIT")),
            "dep_actual":    _parse_ist_time(date_str, row.get("AB_PROGNOSE")),
            "dep_status":    str(row.get("AB_PROGNOSE_STATUS", "")).strip(),
            # Ankunft: geplant und tatsächlich
            "arr_scheduled": _parse_ist_time(date_str, row.get("ANKUNFTSZEIT")),
            "arr_actual":    _parse_ist_time(date_str, row.get("AN_PROGNOSE")),
            "arr_status":    str(row.get("AN_PROGNOSE_STATUS", "")).strip(),
            "cancelled":     cancelled,
        })
        #Umwandlung in einheitliche Spaltennahmen
    return pd.DataFrame(rows)


def build_trip_delays(paths: list[Path]) -> pd.DataFrame:
    """
    Konvertiert Ist-Daten-Dateien in Trainingsdatensätze.

    Pro Fahrt (FAHRT_BEZEICHNER) werden alle Ursprungs-Ziel-Paare erzeugt
    (4 Halte → 6 Segmente). Verspätung = tatsächliche − geplante Ankunft in
    Minuten, negative Werte (Frühankünfte) auf 0 gesetzt.
    Verworfen werden: fehlende Zeiten, Status UNBEKANNT, ausgefallene Fahrten.
    """
    # ── Alle Dateien einlesen und zusammenführen ───────────────────────────────
    frames = []
    print(f"  📂  Verarbeite {len(paths)} Ist-Daten-Dateien …")
    for p in paths:
        df = parse_istdaten_file(p)
        if not df.empty:
            frames.append(df)

    if not frames:
        print("Keine Daten geladen — bitte CSV-Dateien in istdaten_cache/ ablegen.")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    print(f"      {len(raw):,} Halte-Zeilen geladen")

    records = []
    # Nur Einträge mit tatsächlich gemessenen oder prognostizierten Zeiten
    VALID_STATUSES = {"REAL", "PROGNOSE"}

    # ── Über alle Fahrten iterieren ────────────────────────────────────────────
    for (dep_date, fahrt_id), grp in raw.groupby(["date", "fahrt_id"]):
        linien = grp["linien_text"].iloc[0]   # z. B. "EC"

        # Pro Fahrt: eine Map von Haltestellenname → Zeile
        # (erster Eintrag gewinnt, falls eine Haltestelle doppelt vorkommt)
        stop_map: dict[str, pd.Series] = {}
        for _, row in grp.iterrows():
            s = row["stop"]
            if s not in stop_map:
                stop_map[s] = row

        # Nur Fahrten mit mindestens 2 Streckenhaltestellen sind sinnvoll
        corridor = [s for s in STOPS_ORDERED if s in stop_map]
        if len(corridor) < 2:
            continue

        # ── Fahrtrichtung bestimmen ────────────────────────────────────────────
        # Vergleich der geplanten Abfahrtszeiten: erste vs. letzte Haltestelle.
        # Wenn erste < letzte → Zürich → St. Gallen (direction = 0).
        first_row = stop_map[corridor[0]]
        last_row  = stop_map[corridor[-1]]
        t_first   = first_row["dep_scheduled"] or first_row["arr_scheduled"]
        t_last    = last_row["dep_scheduled"]  or last_row["arr_scheduled"]

        if t_first is None or t_last is None:
            continue

        if t_first < t_last:
            direction     = 0   # Zürich HB → St. Gallen
            ordered_stops = [s for s in STOPS_ORDERED if s in stop_map]
        else:
            direction     = 1   # St. Gallen → Zürich HB
            ordered_stops = [s for s in reversed(STOPS_ORDERED) if s in stop_map]

        # ── Alle Ursprungs-Ziel-Paare erzeugen ────────────────────────────────
        for i, origin in enumerate(ordered_stops[:-1]):
            orig_row  = stop_map[origin]
            dep_sched = orig_row["dep_scheduled"]
            if dep_sched is None:
                continue  # Keine Abfahrtszeit → Datensatz nicht verwendbar

            for destination in ordered_stops[i + 1:]:
                dest_row   = stop_map[destination]
                arr_sched  = dest_row["arr_scheduled"]
                arr_actual = dest_row["arr_actual"]
                arr_status = dest_row["arr_status"]

                # Qualitätsprüfungen
                if arr_sched is None or arr_actual is None:
                    continue
                if arr_status not in VALID_STATUSES:
                    continue  # Unbekannte oder fehlende Prognose verwerfen
                if dest_row["cancelled"] or orig_row["cancelled"]:
                    continue  # Ausgefallene Fahrten nicht als Trainingsdaten verwenden

                # Verspätung in Minuten berechnen (nie negativ)
                arr_delay = (arr_actual - arr_sched).total_seconds() / 60
                arr_delay = max(0.0, round(arr_delay, 1))

                orig_idx = STOP_TO_IDX[origin]
                dest_idx = STOP_TO_IDX[destination]

                records.append({
                    "date":                  dep_date,
                    "linien_text":           linien,
                    "direction":             direction,
                    "origin_stop":           origin,
                    "destination_stop":      destination,
                    "origin_stop_idx":       orig_idx,
                    "destination_stop_idx":  dest_idx,
                    # Segmentlänge = Anzahl Haltestellen zwischen Ursprung und Ziel
                    "segment_length":        abs(dest_idx - orig_idx),
                    "dep_scheduled":         dep_sched,
                    "arr_delay_min":         arr_delay,
                    "is_delayed":            int(arr_delay >= DELAY_THRESHOLD_MIN),
                })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    pct = df["is_delayed"].mean()
    print(f"      {len(df):,} Segment-Datensätze  |  verspätet ≥{DELAY_THRESHOLD_MIN} min: {pct:.1%}")
    return df


def _date_from_filename(path: Path) -> date | None:
    """Extrahiert ein Datum aus dem Dateinamen (z. B. 2024-03-15_istdaten.csv).
    Gibt None zurück, wenn kein Datum erkannt wurde."""
    name = path.stem.replace(".csv", "")   # Doppelte Endung ".csv" berücksichtigen
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%d.%m.%Y"):
        # Dateinamen anhand verschiedener Trennzeichen zerlegen und jeden Teil prüfen
        for part in name.split("_") + name.split("-") + [name]:
            try:
                return datetime.strptime(part.strip()[:10], fmt).date()
            except ValueError:
                pass
    return None


def _find_cached_files(start_date: str | None = None,
                       end_date:   str | None = None) -> list[Path]:
    """
    Gibt alle CSV-Dateien aus dem Cache im angegebenen Datumsbereich zurück.
    Dateien ohne erkennbares Datum im Namen werden immer eingeschlossen.
    """
    if not ISTDATEN_CACHE_DIR.exists():
        return []

    start = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    end   = datetime.strptime(end_date,   "%Y-%m-%d").date() if end_date   else None

    files = []
    for p in sorted(ISTDATEN_CACHE_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".csv", ".gz"):
            continue
        d = _date_from_filename(p)
        if d is None:
            files.append(p)   # Unbekanntes Datum → immer einschliessen
        elif start and end and not (start <= d <= end):
            continue          # Ausserhalb des gewünschten Bereichs → überspringen
        else:
            files.append(p)

    if files:
        print(f"  📁  {len(files)} gecachte Datei(en) im Bereich ({start_date} → {end_date}):")
        for f in files:
            d = _date_from_filename(f)
            print(f"       {f.name}  {'(' + d.isoformat() + ')' if d else ''}")
    return files


def load_real_delays(start_date: str, end_date: str) -> pd.DataFrame:
    """Einstiegspunkt: Ist-Daten aus Cache laden und Verspätungsdatensätze zurückgeben."""
    cached = _find_cached_files(start_date, end_date)
    if not cached:
        return pd.DataFrame()
    return build_trip_delays(cached)


# ══════════════════════════════════════════════════════════════════════════════
# ABSCHNITT 2 — WETTERDATEN
# ══════════════════════════════════════════════════════════════════════════════
#
# Für jede Trainingsstichprobe werden die Wetterbedingungen zum Zeitpunkt der
# Abfahrt (am Ursprungsbahnhof) und bei der Ankunft (am Zielbahnhof) benötigt.
# Die Open-Meteo Archive API liefert stündliche Historik für beliebige Koordinaten.
def fetch_weather_archive(stop: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Stündliche historische Wetterdaten für eine Haltestelle von Open-Meteo.
    Rückgabe: DataFrame mit (date, hour) für schnellen Lookup.
    """
    coords = STOP_COORDS[stop]
    resp   = requests.get(OPENMETEO_ARCHIVE, params={
        "latitude":   coords["lat"],
        "longitude":  coords["lon"],
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     WEATHER_VARIABLES,
        "timezone":   "Europe/Zurich",
    }, timeout=30)
    resp.raise_for_status()

    df = pd.DataFrame(resp.json().get("hourly", {}))
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date    # Python date-Objekt für den Index
    df["hour"] = df["time"].dt.hour    # Integer 0–23 für den Index
    return df.set_index(["date", "hour"])


def weather_at(df: pd.DataFrame, dt: datetime) -> dict | None:
    """Wetterdatensatz für eine bestimmte Stunde; None wenn kein Eintrag vorhanden."""
    key = (dt.date(), dt.hour)
    return df.loc[key].to_dict() if key in df.index else None


# ══════════════════════════════════════════════════════════════════════════════
# ABSCHNITT 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
#
# 42 Merkmale: Strecke (4), Zeit (7), Wetter Ursprung (11), Wetter Ziel (11),
# Abgeleitete Wettermerkmale (9). Reihenfolge muss mit der App übereinstimmen.

FEATURE_COLS = [
    # ── Streckenmerkmale ──────────────────────────────────────────────────────
    "direction",              # 0 = Zürich→St.Gallen, 1 = St.Gallen→Zürich
    "origin_stop_idx",        # Position der Abfahrtshaltestelle (0–4)
    "destination_stop_idx",   # Position der Ankunftshaltestelle (0–4)
    "segment_length",         # Anzahl Haltestellen zwischen Ursprung und Ziel (1–3)
    # ── Zeitmerkmale ─────────────────────────────────────────────────────────
    "dep_hour",               # Abfahrtsstunde (0–23)
    "day_of_week",            # Wochentag (0=Mo, 6=So)
    "month",                  # Monat (1–12)
    "week_of_year",           # Kalenderwoche (1–53)
    "is_weekend",             # Flag: Wochenende (Sa/So = 1)
    "is_rush_hour",           # Flag: Hauptverkehrszeit (7–9 oder 16–19 Uhr = 1)
    "season",                 # Jahreszeit (0=Winter, 1=Frühling, 2=Sommer, 3=Herbst)
    # ── Wetter am Ursprungsbahnhof ────────────────────────────────────────────
    "orig_temp",              # Temperatur [°C]
    "orig_precip",            # Niederschlag [mm]
    "orig_snow",              # Schneefall [cm]
    "orig_snow_depth",        # Schneehöhe am Boden [m]
    "orig_wind",              # Windgeschwindigkeit [km/h]
    "orig_gusts",             # Windböen [km/h]
    "orig_visibility",        # Sichtweite [m]
    "orig_cloud",             # Bewölkung [%]
    "orig_humidity",          # Luftfeuchtigkeit [%]
    "orig_pressure",          # Luftdruck [hPa]
    "orig_weather_code",      # WMO-Wettercode
    # ── Wetter am Zielbahnhof ─────────────────────────────────────────────────
    "dest_temp",
    "dest_precip",
    "dest_snow",
    "dest_snow_depth",
    "dest_wind",
    "dest_gusts",
    "dest_visibility",
    "dest_cloud",
    "dest_humidity",
    "dest_pressure",
    "dest_weather_code",
    # ── Abgeleitete Merkmale ──────────────────────────────────────────────────
    "bad_weather_score",      # Kombinierter Schlechtwetter-Score (0–4)
    "extreme_cold",           # Flag: Temperatur < −5 °C (Weichen einfrieren)
    "extreme_heat",           # Flag: Temperatur > 33 °C (Gleisverbiegung)
    "any_snow",               # Flag: Schneefall > 0 cm
    "heavy_rain",             # Flag: Niederschlag > 8 mm (starker Regen)
    "low_visibility",         # Flag: Sicht < 1000 m (Nebel, dichter Schnee)
    "precip_diff",            # Absoluter Unterschied im Niederschlag (Ziel − Ursprung)
    "temp_diff",              # Absoluter Temperaturunterschied (Ziel − Ursprung)
    "wind_diff",              # Absoluter Windunterschied (Ziel − Ursprung)
]


def _wv(w: dict | None, key: str, default=np.nan):
    """Wetterwert mit Fallback; gibt default zurück wenn w=None oder Wert fehlt."""
    if w is None:
        return default
    v = w.get(key)
    return v if v is not None else default


def weather_record(w: dict | None, prefix: str) -> dict:
    """Wandelt ein Open-Meteo-Dict in 11 benannte Merkmale mit Präfix "orig_" oder "dest_" um."""
    return {
        f"{prefix}_temp":         _wv(w, "temperature_2m"),
        f"{prefix}_precip":       _wv(w, "precipitation",     0),   # Kein Regen = 0, nicht NaN
        f"{prefix}_snow":         _wv(w, "snowfall",          0),   # Kein Schnee = 0, nicht NaN
        f"{prefix}_snow_depth":   _wv(w, "snow_depth",        0),
        f"{prefix}_wind":         _wv(w, "wind_speed_10m"),
        f"{prefix}_gusts":        _wv(w, "wind_gusts_10m"),
        f"{prefix}_visibility":   _wv(w, "visibility",    10000),   # Fehlender Wert = gute Sicht
        f"{prefix}_cloud":        _wv(w, "cloud_cover"),
        f"{prefix}_humidity":     _wv(w, "relative_humidity_2m"),
        f"{prefix}_pressure":     _wv(w, "surface_pressure"),
        f"{prefix}_weather_code": _wv(w, "weather_code",      0),   # 0 = Klarer Himmel
    }


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Berechnet 9 abgeleitete Merkmale aus den Basiswetterwerten.

    bad_weather_score (0–4): gewichtete Summe aus Schnee (×1.5), Regen (×1.0),
        Böen (×0.8), Sichtverlust (×0.7) — je auf [0,1] normiert.
    extreme_cold: Temp < −5 °C (Weichen können einfrieren)
    extreme_heat: Temp > 33 °C (Risiko von Gleisverbiegungen/Hitzebeulen)
    heavy_rain:   Niederschlag > 8 mm (Erdrutsch-/Überflutungsrisiko)
    low_visibility: Sicht < 1000 m (dichter Nebel oder Schneetreiben)
    precip_diff / temp_diff / wind_diff: Wettergradienten entlang der Strecke
    """
    df = df.copy()

    # Schlechtwetter-Score: jede Komponente auf [0,1] normiert, dann gewichtet
    # mit dem Clip wird auf einen gewissen wert gecappt, damit Extremwerte die Skala nicht sprengen
    df["bad_weather_score"] = (
          (df["orig_snow"].clip(0, 5)     / 5)       * 1.5
        + (df["orig_precip"].clip(0, 15)  / 15)      * 1.0
        + (df["orig_gusts"].clip(0, 100)  / 100)     * 0.8
        + ((10_000 - df["orig_visibility"].clip(0, 10_000)) / 10_000) * 0.7
    )

    # Binäre Flags für spezifische Wetterextreme
    df["extreme_cold"]   = (df["orig_temp"]       < -5   ).astype(int)
    df["extreme_heat"]   = (df["orig_temp"]       > 33   ).astype(int)
    df["any_snow"]       = (df["orig_snow"]       > 0    ).astype(int)
    df["heavy_rain"]     = (df["orig_precip"]     > 8    ).astype(int)
    df["low_visibility"] = (df["orig_visibility"] < 1_000).astype(int)

    # Differenzmerkmale: Wettergradienten entlang der Strecke
    df["precip_diff"] = (df["dest_precip"] - df["orig_precip"]).abs()
    df["temp_diff"]   = (df["dest_temp"]   - df["orig_temp"]  ).abs()
    df["wind_diff"]   = (df["dest_wind"]   - df["orig_wind"]  ).abs()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# ABSCHNITT 4 — DATENSATZ AUFBAUEN
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Vollständige Merkmalsmatrix aus Ist-Daten und Wetterdaten erstellen.

    Ankunftswetter wird mit Abfahrt + 1h06min approximiert (Gesamtfahrzeit
    ZH HB → SG); für kürzere Segmente leicht überschätzt, der Stunden-Index
    bleibt aber korrekt.
    """
    print(f"\n📅  Datensatz aufbauen  {start_date} → {end_date}")

    # ── Schritt 1: Ist-Daten laden ─────────────────────────────────────────────
    print("\n🚆  Lade Ist-Daten …")
    df_delays = load_real_delays(start_date, end_date)
    if df_delays.empty:
        raise SystemExit(
            "\nKeine Verspätungsdaten geladen.\n"
            "Bitte Ist-Daten-CSV-Dateien manuell herunterladen und in\n"
            "den Ordner  istdaten_cache/  ablegen.\n\n"
            "Download: https://opentransportdata.swiss/de/dataset/istdaten"
        )

    # ── Schritt 2: Wetterdaten laden ───────────────────────────────────────────
    print(f"\n🌤️  Lade Wetterdaten für {len(STOPS_ORDERED)} Haltestellen …")
    weather_cache: dict[str, pd.DataFrame] = {}
    for stop in STOPS_ORDERED:
        print(f"    {stop} …", end=" ", flush=True)
        weather_cache[stop] = fetch_weather_archive(stop, start_date, end_date)
        print(f"✓ ({len(weather_cache[stop]):,} Stunden)")
        time.sleep(0.4)   # Kurze Pause, um die Open-Meteo API nicht zu überlasten

    # ── Schritt 3: Merkmale zusammenführen ────────────────────────────────────
    print("\n🔧  Erstelle Merkmalsvektoren …")
    records = []
    for _, trip in df_delays.iterrows():
        dep_dt = trip["dep_scheduled"]
        if not isinstance(dep_dt, datetime):
            dep_dt = pd.Timestamp(dep_dt).to_pydatetime()

        # Ankunftszeit approximieren: +1h06min (volle Fahrtdauer ZH → SG)
        arr_dt = dep_dt + timedelta(hours=1, minutes=6)

        orig_stop = trip["origin_stop"]
        dest_stop = trip["destination_stop"]

        # Wetter zum Abfahrtszeitpunkt am Ursprungsbahnhof
        w_orig = weather_at(weather_cache[orig_stop], dep_dt)
        # Wetter zur (approx.) Ankunftszeit am Zielbahnhof
        w_dest = weather_at(weather_cache[dest_stop], arr_dt)

        month        = dep_dt.month
        dep_hour     = dep_dt.hour
        day_of_week  = dep_dt.weekday()
        week_of_year = dep_dt.isocalendar().week

        records.append({
            # Streckenmerkmale
            "direction":              trip["direction"],
            "origin_stop_idx":        trip["origin_stop_idx"],
            "destination_stop_idx":   trip["destination_stop_idx"],
            "segment_length":         trip["segment_length"],
            # Zeitmerkmale
            "dep_hour":     dep_hour,
            "day_of_week":  day_of_week,
            "month":        month,
            "week_of_year": week_of_year,
            "is_weekend":   int(day_of_week >= 5),
            "is_rush_hour": int(7 <= dep_hour <= 9 or 16 <= dep_hour <= 19),
            "season":       (month % 12) // 3,   # 0=Winter, 1=Frühling, 2=Sommer, 3=Herbst
            # Wetterdaten (11 Merkmale je Haltestelle)
            **weather_record(w_orig, "orig"),
            **weather_record(w_dest, "dest"),
            # Zielgrössen
            "arr_delay_min": trip["arr_delay_min"],
            "is_delayed":    trip["is_delayed"],
        })

    # Datensätze ohne Wetterdaten verwerfen (kein Wettereintrag für diese Stunde)
    df = pd.DataFrame(records).dropna(subset=["orig_temp"])
    pct = df["is_delayed"].mean()
    print(f"    ✓ {len(df):,} Stichproben  |  verspätet ≥{DELAY_THRESHOLD_MIN} min: {pct:.1%}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ABSCHNITT 5 — MODELLTRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_and_save(df: pd.DataFrame) -> None:
    """
    Klassifikator und Regressor trainieren und als ec_models.pkl speichern.

    Hyperparameter-Begründung:
      n_estimators=300:   Mehr Bäume stabilisieren die Vorhersage; ab ~300
                          sind die Verbesserungen marginal bei vertretbarem Rechenaufwand.
      max_depth=14:       Verhindert Overfitting — zu tiefe Bäume lernen Rauschen,
                          zu flache verpassen komplexe Muster.
      min_samples_leaf=5: Mindestgrösse jedes Blattknotens, verhindert das
                          Auswendiglernen einzelner Ausreisser.
      class_weight='balanced': Gleicht unbalancierte Klassen aus — da verspätete
                          Züge die Minderheit sind, würde das Modell ohne diesen
                          Parameter fast immer "pünktlich" vorhersagen.

    Train/Test-Split: 80/20, Klassifikator stratifiziert (gleiche Klassenverteilung
    in beiden Splits). Regressor ohne Stratifizierung (kontinuierliche Zielgrösse).
    """
    # Abgeleitete Merkmale berechnen
    df = add_derived_features(df)
    X  = df[FEATURE_COLS].fillna(0)   # Fehlende Wetterwerte mit 0 auffüllen

    # ── Klassifikator ─────────────────────────────────────────────────────────
    print("\n🌲  Trainiere Klassifikator (pünktlich vs. verspätet ≥3 min) …")
    y_cls = df["is_delayed"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=5,
        class_weight="balanced",   # Ausgleich für unbalancierte Klassen
        n_jobs=-1,                  # Alle CPU-Kerne nutzen
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    print(classification_report(
        y_te, clf.predict(X_te),
        target_names=["Pünktlich", "Verspätet ≥3 min"],
    ))

    # ── Regressor ─────────────────────────────────────────────────────────────
    print("🌲  Trainiere Regressor (Verspätung in Minuten) …")
    y_reg = df["arr_delay_min"]
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    reg.fit(X_tr2, y_tr2)
    y_pred = reg.predict(X_te2)
    print(f"    MAE: {mean_absolute_error(y_te2, y_pred):.2f} min  |  "
          f"R²: {r2_score(y_te2, y_pred):.3f}")

    # ── Merkmalswichtigkeit ───────────────────────────────────────────────────
    # Gini-Wichtigkeit: zeigt, wie oft ein Merkmal für Splits verwendet wurde
    # und wie stark es die Unreinheit (Gini-Koeffizient) reduziert hat.
    imp = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n🔍  Top 12 Merkmale (Klassifikator):")
    for feat, val in imp.head(12).items():
        print(f"  {feat:<30} {val:.4f}  {'█' * max(1, int(val * 70))}")

    # ── Modelle und Metadaten speichern ───────────────────────────────────────
    joblib.dump({
        "clf":          clf,
        "reg":          reg,
        "feature_cols": FEATURE_COLS,
        "stops":        STOPS_ORDERED,
        "stop_to_idx":  STOP_TO_IDX,
        "stop_coords":  STOP_COORDS,
        "threshold":    DELAY_THRESHOLD_MIN,
    }, MODEL_PATH)
    print(f"\n💾  Modelle gespeichert → {MODEL_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# EINSTIEGSPUNKT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "EC-Verspätungsprediktor — Trainingsskript.\n"
            "Ist-Daten-CSV-Dateien müssen im Ordner istdaten_cache/ liegen."
        )
    )
    parser.add_argument(
        "--start",
        default="2026-01-01",
        help="Startdatum des Trainingszeitraums (YYYY-MM-DD, Standard: 2026-01-01)",
    )
    parser.add_argument(
        "--end",
        default="2026-03-31",
        help="Enddatum des Trainingszeitraums (YYYY-MM-DD, Standard: 2026-03-31)",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("  EC-Verspätungsprediktor  |  Training")
    print("=" * 62)
    print(f"  Zeitraum: {args.start} → {args.end}")
    print(f"  Cache-Ordner: {ISTDATEN_CACHE_DIR.resolve()}")
    print()

    # Datensatz aufbauen und speichern
    df = build_dataset(args.start, args.end)
    df.to_csv("ec_delay_dataset.csv", index=False)
    print(f"💾  Datensatz gespeichert → ec_delay_dataset.csv")

    # Modelle trainieren und speichern
    train_and_save(df)
    print("\n✅  Training abgeschlossen.  Starten Sie die App mit:  streamlit run app.py")
