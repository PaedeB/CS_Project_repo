import streamlit as st
import pandas as pd
import numpy as np


import requests
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


import json
import time

print("test")


## Funktioniert nicht

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# 🚆 SBB Strecken-Zuverlässigkeits-Dashboard
# Gruppenprojekt – Modul: Computer Science
# ============================================================
# Benötigte Installationen (einmalig im Terminal ausführen):
#   pip install streamlit pandas plotly scikit-learn requests numpy
#
# App starten:
#   streamlit run sbb_dashboard.py
# ============================================================
#
# VERWENDETE APIs (alle kostenlos, kein API-Key nötig):
#   1. transport.opendata.ch  → Echtzeit-Verbindungen & Verspätungen
#   2. data.sbb.ch            → Aktuelle SBB Störungsmeldungen (RSS-Feed)
#   3. open-meteo.com         → Wetterdaten (historisch & aktuell)
#
# ANALYSIERTE STRECKE: Bern ↔ Zürich HB
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
import time

# ============================================================
# KONFIGURATION
# Zentrale Einstellungen – hier anpassen falls nötig
# ============================================================

# Analysierte Strecke (kann später angepasst werden)
STRECKE = {
    "name": "Bern → Zürich HB",
    "von": "Bern",
    "nach": "Zürich HB",
    "von_id": "8507000",   # SBB Stations-ID Bern
    "nach_id": "8503000",  # SBB Stations-ID Zürich HB
}

# API Endpoints
API_TRANSPORT   = "https://transport.opendata.ch/v1"
API_STOERUNGEN  = "https://www.sbb.ch/content/dam/internet/sbb/de/meta/footer/baustellen.xml"  # SBB Störungs-RSS
API_WETTER      = "https://api.open-meteo.com/v1/forecast"
API_WETTER_HIST = "https://archive-api.open-meteo.com/v1/archive"

# Koordinaten Bern (für Wetter-API)
BERN_LAT = 46.9480
BERN_LON = 7.4474

# ============================================================
# DATENBANK SETUP
# Speichert historische Abfragen lokal für ML-Training
# ============================================================

def init_db():
    """
    Initialisiert die SQLite Datenbank.
    Erstellt zwei Tabellen:
      - verspaetungen: Gespeicherte Verspätungsmessungen pro Fahrt
      - stoerungen:    Gespeicherte Störungsmeldungen
    """
    conn = sqlite3.connect("sbb_data.db")
    cursor = conn.cursor()

    # Tabelle für Verspätungsdaten
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS verspaetungen (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datum TEXT,
            wochentag INTEGER,        -- 0=Montag, 6=Sonntag
            stunde INTEGER,           -- Abfahrtsstunde (0-23)
            verspaetung_min REAL,     -- Verspätung in Minuten
            ist_verspaetet INTEGER,   -- 1 = ja (>3 min), 0 = nein
            temperatur REAL,          -- °C
            niederschlag REAL,        -- mm
            schneefall REAL,          -- cm
            windgeschwindigkeit REAL  -- km/h
        )
    """)

    # Tabelle für Störungsmeldungen
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stoerungen (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datum TEXT,
            titel TEXT,
            beschreibung TEXT,
            strecke TEXT
        )
    """)

    conn.commit()
    conn.close()

def verspaetung_speichern(datum, wochentag, stunde, verspaetung_min,
                          temperatur, niederschlag, schneefall, wind):
    """Speichert einen Verspätungseintrag in der Datenbank."""
    conn = sqlite3.connect("sbb_data.db")
    cursor = conn.cursor()
    ist_verspaetet = 1 if verspaetung_min > 3 else 0  # >3 Min gilt als verspätet
    cursor.execute("""
        INSERT INTO verspaetungen
        (datum, wochentag, stunde, verspaetung_min, ist_verspaetet,
         temperatur, niederschlag, schneefall, windgeschwindigkeit)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (datum, wochentag, stunde, verspaetung_min, ist_verspaetet,
          temperatur, niederschlag, schneefall, wind))
    conn.commit()
    conn.close()

def daten_laden():
    """Lädt alle gespeicherten Verspätungsdaten aus der Datenbank."""
    conn = sqlite3.connect("sbb_data.db")
    df = pd.read_sql("SELECT * FROM verspaetungen", conn)
    conn.close()
    return df

def stoerungen_laden_db():
    """Lädt alle gespeicherten Störungsmeldungen aus der Datenbank."""
    conn = sqlite3.connect("sbb_data.db")
    df = pd.read_sql("SELECT * FROM stoerungen ORDER BY datum DESC", conn)
    conn.close()
    return df

# ============================================================
# API: TRANSPORT.OPENDATA.CH
# Lädt aktuelle Verbindungen und Verspätungen
# ============================================================

@st.cache_data(ttl=120)  # 2 Minuten Cache (Echtzeit-Daten)
def verbindungen_laden(von, nach, anzahl=10):
    """
    Ruft aktuelle Zugverbindungen von transport.opendata.ch ab.
    Gibt einen DataFrame mit Verbindungen inkl. Verspätungen zurück.

    Dokumentation: https://transport.opendata.ch/docs.html
    """
    try:
        params = {
            "from": von,
            "to": nach,
            "limit": anzahl,
            "fields[]": ["connections/from", "connections/to",
                         "connections/duration", "connections/products"]
        }
        response = requests.get(f"{API_TRANSPORT}/connections", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        verbindungen = []
        for conn in data.get("connections", []):
            abfahrt_geplant = conn["from"].get("departure", "")
            abfahrt_ist     = conn["from"].get("prognosis", {}).get("departure", "")
            ankunft_geplant = conn["to"].get("arrival", "")
            ankunft_ist     = conn["to"].get("prognosis", {}).get("arrival", "")

            # Verspätung berechnen (falls Echtzeit-Daten vorhanden)
            verspaetung = 0
            if abfahrt_geplant and abfahrt_ist:
                try:
                    geplant_dt = datetime.fromisoformat(abfahrt_geplant[:19])
                    ist_dt     = datetime.fromisoformat(abfahrt_ist[:19])
                    verspaetung = max(0, (ist_dt - geplant_dt).seconds // 60)
                except:
                    verspaetung = 0

            verbindungen.append({
                "Abfahrt (geplant)": abfahrt_geplant[:16] if abfahrt_geplant else "–",
                "Abfahrt (ist)":     abfahrt_ist[:16]     if abfahrt_ist     else "–",
                "Ankunft (geplant)": ankunft_geplant[:16] if ankunft_geplant else "–",
                "Verspätung (min)":  verspaetung,
                "Status": "🔴 Verspätet" if verspaetung > 3 else "🟢 Pünktlich"
            })

        return pd.DataFrame(verbindungen)

    except requests.exceptions.RequestException as e:
        st.warning(f"Verbindungsfehler zur Transport-API: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=120)
def abfahrten_laden(station, anzahl=20):
    """
    Lädt aktuelle Abfahrten an einem Bahnhof.
    Nützlich für das Echtzeit-Dashboard.
    """
    try:
        params = {"id": station, "limit": anzahl}
        response = requests.get(f"{API_TRANSPORT}/stationboard", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        abfahrten = []
        for journey in data.get("stationboard", []):
            stop = journey.get("stop", {})
            abfahrt    = stop.get("departure", "")
            prognose   = stop.get("prognosis", {}).get("departure", "")
            verspaetung = 0
            if abfahrt and prognose:
                try:
                    a_dt = datetime.fromisoformat(abfahrt[:19])
                    p_dt = datetime.fromisoformat(prognose[:19])
                    verspaetung = max(0, (p_dt - a_dt).seconds // 60)
                except:
                    pass

            abfahrten.append({
                "Zug":             journey.get("number", "–"),
                "Richtung":        journey.get("to", "–"),
                "Abfahrt":         abfahrt[11:16] if abfahrt else "–",
                "Verspätung (min)": verspaetung,
                "Status":          "🔴 Verspätet" if verspaetung > 3 else "🟢 Pünktlich"
            })

        return pd.DataFrame(abfahrten)

    except requests.exceptions.RequestException as e:
        st.warning(f"Fehler beim Laden der Abfahrten: {e}")
        return pd.DataFrame()

# ============================================================
# API: SBB STÖRUNGSMELDUNGEN
# Lädt aktuelle Störungen als RSS-Feed
# ============================================================

@st.cache_data(ttl=300)  # 5 Minuten Cache
def stoerungen_laden_api():
    """
    Lädt aktuelle SBB-Störungsmeldungen.

    TODO für die Gruppe: Den korrekten RSS-Feed-Link der SBB einfügen.
    Alternativ kann auch die offizielle SBB-Störungsseite gescrapt werden.
    Überprüft auf: https://www.sbb.ch/de/kaufen/pages/fahrplan/aktuell.xhtml
    """
    try:
        response = requests.get(API_STOERUNGEN, timeout=10)
        response.raise_for_status()

        # XML parsen
        root = ET.fromstring(response.content)
        stoerungen = []
        for item in root.findall(".//item"):
            titel = item.findtext("title", "")
            beschr = item.findtext("description", "")
            datum  = item.findtext("pubDate", "")
            stoerungen.append({
                "Datum":        datum,
                "Störung":      titel,
                "Beschreibung": beschr
            })
        return pd.DataFrame(stoerungen)

    except Exception as e:
        # Fallback: Leere Liste mit Hinweis
        # (API-Link muss noch verifiziert werden)
        return pd.DataFrame(columns=["Datum", "Störung", "Beschreibung"])

# ============================================================
# API: OPEN-METEO
# Lädt Wetterdaten (historisch und aktuell)
# ============================================================

@st.cache_data(ttl=3600)  # 1 Stunde Cache
def wetter_aktuell_laden():
    """
    Lädt aktuelle Wetterdaten für Bern.
    Gibt Temperatur, Niederschlag, Schneefall und Wind zurück.
    Dokumentation: https://open-meteo.com/en/docs
    """
    try:
        params = {
            "latitude":  BERN_LAT,
            "longitude": BERN_LON,
            "current":   "temperature_2m,precipitation,snowfall,wind_speed_10m",
            "timezone":  "Europe/Zurich"
        }
        response = requests.get(API_WETTER, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("current", {})

    except Exception as e:
        st.warning(f"Wetter-API nicht erreichbar: {e}")
        return {}

@st.cache_data(ttl=86400)  # 24 Stunden Cache (historische Daten ändern sich nicht)
def wetter_historisch_laden(start_datum, end_datum):
    """
    Lädt historische stündliche Wetterdaten.
    start_datum / end_datum: Format "YYYY-MM-DD"
    Nützlich für ML-Training.
    """
    try:
        params = {
            "latitude":   BERN_LAT,
            "longitude":  BERN_LON,
            "start_date": start_datum,
            "end_date":   end_datum,
            "hourly":     "temperature_2m,precipitation,snowfall,wind_speed_10m",
            "timezone":   "Europe/Zurich"
        }
        response = requests.get(API_WETTER_HIST, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # In DataFrame umwandeln
        df = pd.DataFrame({
            "Datum":             data["hourly"]["time"],
            "Temperatur (°C)":   data["hourly"]["temperature_2m"],
            "Niederschlag (mm)": data["hourly"]["precipitation"],
            "Schneefall (cm)":   data["hourly"]["snowfall"],
            "Wind (km/h)":       data["hourly"]["wind_speed_10m"]
        })
        df["Datum"] = pd.to_datetime(df["Datum"])
        return df

    except Exception as e:
        st.warning(f"Historische Wetterdaten nicht ladbar: {e}")
        return pd.DataFrame()

# ============================================================
# MASCHINELLES LERNEN
# Random Forest Classifier für Verspätungsrisiko
# ============================================================

def trainings_daten_generieren(n=500):
    """
    Generiert synthetische Trainingsdaten bis echte historische
    Daten aus der Datenbank verfügbar sind.

    WICHTIG: Diese Funktion durch echte Daten ersetzen sobald
    genug Datenpunkte in der DB gesammelt wurden (ca. 2-3 Wochen)!

    Muster (basierend auf bekannten SBB-Mustern):
    - Rushhour (7-9h, 17-19h) → mehr Verspätungen
    - Schneefall > 2cm        → hohes Risiko
    - Temperatur < 0°C        → erhöhtes Risiko
    - Wochenende              → weniger Verspätungen
    """
    np.random.seed(42)
    n_samples = n

    wochentag   = np.random.randint(0, 7, n_samples)
    stunde      = np.random.randint(5, 23, n_samples)
    temperatur  = np.random.uniform(-10, 35, n_samples)
    niederschlag = np.random.uniform(0, 20, n_samples)
    schneefall  = np.random.uniform(0, 15, n_samples)
    wind        = np.random.uniform(0, 80, n_samples)

    # Risiko-Logik für synthetische Labels
    risiko = []
    for i in range(n_samples):
        score = 0
        # Rushhour
        if stunde[i] in [7, 8, 9, 17, 18, 19]:    score += 2
        # Schnee
        if schneefall[i] > 5:                       score += 3
        elif schneefall[i] > 2:                     score += 1
        # Kälte
        if temperatur[i] < -5:                      score += 2
        elif temperatur[i] < 0:                     score += 1
        # Starker Wind
        if wind[i] > 60:                            score += 2
        elif wind[i] > 40:                          score += 1
        # Starker Regen
        if niederschlag[i] > 10:                    score += 1
        # Wochenende (weniger Betrieb → weniger Verspätungen)
        if wochentag[i] in [5, 6]:                  score -= 1

        if score <= 1:   risiko.append(0)   # Niedrig
        elif score <= 3: risiko.append(1)   # Mittel
        else:            risiko.append(2)   # Hoch

    df = pd.DataFrame({
        "wochentag":        wochentag,
        "stunde":           stunde,
        "temperatur":       temperatur,
        "niederschlag":     niederschlag,
        "schneefall":       schneefall,
        "windgeschwindigkeit": wind,
        "risiko":           risiko
    })
    return df

@st.cache_resource  # Modell wird nur einmal trainiert und gecacht
def ml_modell_trainieren():
    """
    Trainiert den Random Forest Classifier.
    Versucht zuerst echte Daten aus der DB zu laden.
    Falls zu wenig Daten vorhanden, werden synthetische verwendet.
    Gibt das trainierte Modell und die Feature-Namen zurück.
    """
    # Echte Daten versuchen zu laden
    echte_daten = daten_laden()

    if len(echte_daten) >= 100:
        # Genug echte Daten vorhanden!
        df = echte_daten.copy()
        # Risiko-Label aus Verspätungsminuten ableiten
        df["risiko"] = df["verspaetung_min"].apply(
            lambda x: 0 if x <= 3 else (1 if x <= 10 else 2)
        )
    else:
        # Fallback: Synthetische Daten
        df = trainings_daten_generieren(500)

    # Features definieren (was das Modell als Input bekommt)
    features = ["wochentag", "stunde", "temperatur",
                "niederschlag", "schneefall", "windgeschwindigkeit"]

    X = df[features]
    y = df["risiko"]

    # Modell trainieren
    modell = RandomForestClassifier(
        n_estimators=100,    # 100 Entscheidungsbäume
        max_depth=8,         # Tiefe begrenzen (verhindert Overfitting)
        random_state=42
    )
    modell.fit(X, y)

    return modell, features

def risiko_vorhersagen(modell, features, wochentag, stunde,
                       temperatur, niederschlag, schneefall, wind):
    """
    Sagt das Verspätungsrisiko für gegebene Bedingungen voraus.
    Gibt zurück: Risikolevel (0/1/2), Label, Wahrscheinlichkeiten,
                 und Feature Importances.
    """
    # Input als DataFrame (Modell erwartet dieses Format)
    X = pd.DataFrame([[wochentag, stunde, temperatur,
                        niederschlag, schneefall, wind]],
                     columns=features)

    # Vorhersage
    risiko_num    = modell.predict(X)[0]
    wahrscheinl   = modell.predict_proba(X)[0]

    # Feature Importance (welche Faktoren sind am wichtigsten?)
    importances = dict(zip(features, modell.feature_importances_))

    # Numerisches Label in Text umwandeln
    labels = {0: "🟢 Niedrig", 1: "🟡 Mittel", 2: "🔴 Hoch"}
    return risiko_num, labels[risiko_num], wahrscheinl, importances

# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def wochentag_name(nr):
    """Wandelt Wochentag-Nummer in deutschen Namen um."""
    namen = ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
             "Freitag", "Samstag", "Sonntag"]
    return namen[nr]

def risiko_farbe(risiko_label):
    """Gibt die passende Farbe für ein Risikolevel zurück."""
    if "Niedrig" in risiko_label: return "success"
    if "Mittel"  in risiko_label: return "warning"
    return "error"

# ============================================================
# STREAMLIT APP – HAUPTSTRUKTUR
# ============================================================

def main():
    st.set_page_config(
        page_title="🚆 SBB Zuverlässigkeits-Dashboard",
        page_icon="🚆",
        layout="wide"
    )

    # Datenbank beim Start initialisieren
    init_db()

    # Titel mit Streckeninfo
    st.title("🚆 SBB Strecken-Zuverlässigkeit")
    st.caption(f"Analyse der Strecke **{STRECKE['name']}** – Keine offizielle SBB-App.")

    # ---- NAVIGATION ----
    st.sidebar.title("Navigation")
    seite = st.sidebar.radio("Seite wählen:", [
        "📡 Live-Dashboard",
        "📊 Historische Analyse",
        "🤖 Störungsprediktor",
        "ℹ️ Über diese App"
    ])

    # Aktuelles Wetter immer in Sidebar anzeigen
    st.sidebar.divider()
    st.sidebar.subheader("🌤️ Aktuelles Wetter (Bern)")
    wetter = wetter_aktuell_laden()
    if wetter:
        st.sidebar.metric("Temperatur",  f"{wetter.get('temperature_2m', '–')} °C")
        st.sidebar.metric("Niederschlag", f"{wetter.get('precipitation', '–')} mm")
        st.sidebar.metric("Schneefall",  f"{wetter.get('snowfall', '–')} cm")
        st.sidebar.metric("Wind",        f"{wetter.get('wind_speed_10m', '–')} km/h")

    # ============================================================
    # SEITE 1: LIVE-DASHBOARD
    # Zeigt Echtzeit-Verbindungen und aktuelle Störungen
    # ============================================================
    if seite == "📡 Live-Dashboard":
        st.header("📡 Live-Dashboard")
        st.write(f"Aktuelle Verbindungen und Störungen auf der Strecke **{STRECKE['name']}**.")

        # Zwei Spalten: Verbindungen links, Störungen rechts
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("🚆 Nächste Verbindungen")
            with st.spinner("Lade Echtzeit-Daten..."):
                verbindungen = verbindungen_laden(STRECKE["von"], STRECKE["nach"])

            if not verbindungen.empty:
                # Verspätete Zeilen rot einfärben
                st.dataframe(
                    verbindungen,
                    use_container_width=True,
                    hide_index=True
                )
                # Pünktlichkeitsquote berechnen
                puenktlich = (verbindungen["Verspätung (min)"] <= 3).sum()
                total      = len(verbindungen)
                st.metric("Pünktlichkeit (aktuell)",
                          f"{puenktlich}/{total} Züge",
                          f"{(puenktlich/total*100):.0f}%")
            else:
                st.info("Keine Echtzeit-Daten verfügbar.")

        with col2:
            st.subheader("⚠️ Aktuelle Störungen")
            stoerungen = stoerungen_laden_api()
            if not stoerungen.empty:
                for _, row in stoerungen.iterrows():
                    st.warning(f"**{row['Störung']}**\n\n{row['Beschreibung']}")
            else:
                st.success("✅ Keine aktuellen Störungen gemeldet.")

        # Aktueller Daten-Refresh Button
        if st.button("🔄 Daten aktualisieren"):
            st.cache_data.clear()
            st.rerun()

    # ============================================================
    # SEITE 2: HISTORISCHE ANALYSE
    # Zeigt gespeicherte Verspätungsdaten und Wetterzusammenhänge
    # ============================================================
    elif seite == "📊 Historische Analyse":
        st.header("📊 Historische Zuverlässigkeit")
        st.write("Analyse gespeicherter Verspätungsdaten und deren Zusammenhang mit Wetter.")

        df = daten_laden()

        if df.empty:
            st.info("""
            📋 **Noch keine historischen Daten vorhanden.**

            Die App sammelt automatisch Daten, wenn du das Live-Dashboard verwendest.
            Nach einigen Tagen erscheinen hier die Auswertungen.

            *Tipp für die Gruppe: Ihr könnt auch historische SBB-Daten von
            [opentransportdata.swiss](https://opentransportdata.swiss) herunterladen
            und importieren, um sofort Daten zu haben.*
            """)
        else:
            # Kennzahlen
            m1, m2, m3 = st.columns(3)
            m1.metric("Erfasste Fahrten",   len(df))
            m2.metric("Ø Verspätung",        f"{df['verspaetung_min'].mean():.1f} min")
            m3.metric("Pünktlichkeitsquote", f"{(df['ist_verspaetet']==0).mean()*100:.1f}%")

            # Verspätungen nach Stunde
            st.subheader("Verspätungen nach Tageszeit")
            stunden_df = df.groupby("stunde")["verspaetung_min"].mean().reset_index()
            fig1 = px.bar(stunden_df, x="stunde", y="verspaetung_min",
                         labels={"stunde": "Stunde", "verspaetung_min": "Ø Verspätung (min)"},
                         color="verspaetung_min", color_continuous_scale="RdYlGn_r",
                         template="plotly_dark")
            st.plotly_chart(fig1, use_container_width=True)

            # Verspätungen nach Wochentag
            st.subheader("Verspätungen nach Wochentag")
            wt_df = df.groupby("wochentag")["verspaetung_min"].mean().reset_index()
            wt_df["Tag"] = wt_df["wochentag"].apply(wochentag_name)
            fig2 = px.bar(wt_df, x="Tag", y="verspaetung_min",
                         labels={"verspaetung_min": "Ø Verspätung (min)"},
                         template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

            # Zusammenhang Schneefall / Verspätung
            if "schneefall" in df.columns:
                st.subheader("Wetterzusammenhang: Schneefall & Verspätung")
                fig3 = px.scatter(df, x="schneefall", y="verspaetung_min",
                                 color="ist_verspaetet",
                                 labels={"schneefall": "Schneefall (cm)",
                                         "verspaetung_min": "Verspätung (min)"},
                                 template="plotly_dark",
                                 color_discrete_map={0: "#4CAF50", 1: "#F44336"})
                st.plotly_chart(fig3, use_container_width=True)

        # Historische Wetterdaten laden (letzter Monat)
        st.subheader("🌦️ Historisches Wetter (letzter Monat)")
        heute     = datetime.now()
        vor_monat = heute - timedelta(days=30)
        wetter_hist = wetter_historisch_laden(
            vor_monat.strftime("%Y-%m-%d"),
            heute.strftime("%Y-%m-%d")
        )
        if not wetter_hist.empty:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=wetter_hist["Datum"],
                y=wetter_hist["Temperatur (°C)"],
                name="Temperatur (°C)",
                line=dict(color="#2196F3")
            ))
            fig4.add_trace(go.Bar(
                x=wetter_hist["Datum"],
                y=wetter_hist["Schneefall (cm)"],
                name="Schneefall (cm)",
                marker_color="#90CAF9",
                yaxis="y2"
            ))
            fig4.update_layout(
                title="Temperatur & Schneefall – letzter Monat",
                yaxis2=dict(overlaying="y", side="right"),
                template="plotly_dark"
            )
            st.plotly_chart(fig4, use_container_width=True)

    # ============================================================
    # SEITE 3: STÖRUNGSPREDIKTOR
    # ML-Modell sagt Verspätungsrisiko für geplante Reise voraus
    # ============================================================
    elif seite == "🤖 Störungsprediktor":
        st.header("🤖 Störungsprediktor")
        st.write(f"Wie hoch ist das Verspätungsrisiko auf der Strecke **{STRECKE['name']}** für deine Reise?")
        st.warning("⚠️ Diese Vorhersage basiert auf statistischen Mustern und ist keine Garantie.")

        # Modell laden
        with st.spinner("Lade ML-Modell..."):
            modell, features = ml_modell_trainieren()

        # Benutzereingabe
        st.subheader("🗓️ Reise planen")
        col1, col2 = st.columns(2)

        with col1:
            wochentag_input = st.selectbox(
                "Reisetag:",
                options=list(range(7)),
                format_func=wochentag_name
            )
            stunde_input = st.slider("Abfahrtszeit:", 5, 23, 8,
                                     format="%d:00 Uhr")

        with col2:
            # Aktuelles Wetter als Standardwerte
            temp_default  = wetter.get("temperature_2m", 10)  if wetter else 10
            wind_default  = wetter.get("wind_speed_10m", 15)  if wetter else 15
            schnee_default = wetter.get("snowfall", 0)         if wetter else 0
            regen_default  = wetter.get("precipitation", 0)   if wetter else 0

            temp_input   = st.number_input("Temperatur (°C):",  -20.0, 40.0, float(temp_default),  0.5)
            schnee_input = st.number_input("Schneefall (cm):",    0.0, 50.0, float(schnee_default), 0.5)
            regen_input  = st.number_input("Niederschlag (mm):",  0.0, 50.0, float(regen_default),  0.5)
            wind_input   = st.number_input("Wind (km/h):",         0.0, 120.0, float(wind_default),  1.0)

        if st.button("🔍 Risiko berechnen", type="primary"):
            risiko_num, risiko_label, wahrscheinl, importances = risiko_vorhersagen(
                modell, features,
                wochentag_input, stunde_input,
                temp_input, regen_input, schnee_input, wind_input
            )

            # Ergebnis anzeigen
            st.divider()
            st.subheader("Ergebnis")

            # Grosses Risikolevel
            farbe = risiko_farbe(risiko_label)
            if farbe == "success": st.success(f"## {risiko_label}")
            elif farbe == "warning": st.warning(f"## {risiko_label}")
            else: st.error(f"## {risiko_label}")

            # Wahrscheinlichkeiten als Balkendiagramm
            col_a, col_b = st.columns(2)
            with col_a:
                labels_text = ["🟢 Niedrig", "🟡 Mittel", "🔴 Hoch"]
                fig_prob = go.Figure(go.Bar(
                    x=labels_text,
                    y=[w * 100 for w in wahrscheinl],
                    marker_color=["#4CAF50", "#FF9800", "#F44336"]
                ))
                fig_prob.update_layout(
                    title="Wahrscheinlichkeiten (%)",
                    yaxis_title="%",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_prob, use_container_width=True)

            with col_b:
                # Feature Importance
                imp_df = pd.DataFrame({
                    "Faktor": list(importances.keys()),
                    "Wichtigkeit": list(importances.values())
                }).sort_values("Wichtigkeit", ascending=True)

                fig_imp = go.Figure(go.Bar(
                    x=imp_df["Wichtigkeit"],
                    y=imp_df["Faktor"],
                    orientation="h",
                    marker_color="#2196F3"
                ))
                fig_imp.update_layout(
                    title="Wichtigste Einflussfaktoren",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_imp, use_container_width=True)

    # ============================================================
    # SEITE 4: ÜBER DIESE APP
    # ============================================================
    elif seite == "ℹ️ Über diese App":
        st.header("ℹ️ Über diese App")

        st.markdown(f"""
        ### 🚆 SBB Strecken-Zuverlässigkeits-Dashboard

        **Problem:** Reisende erfahren SBB-Störungen meist erst, wenn sie bereits passiert sind.
        Es gibt keine öffentliche Möglichkeit, proaktiv zu prüfen, wie zuverlässig eine Strecke
        historisch war.

        **Unsere Lösung:** Eine Anwendung, die SBB-Echtzeitdaten, historische Verspätungsstatistiken
        und Wetterdaten kombiniert, um Pendlern eine datenbasierte Antwort zu geben:
        *"Wie zuverlässig ist meine Strecke heute?"*

        ---

        ### 📡 Verwendete Datenquellen
        | Quelle | Daten | Link |
        |---|---|---|
        | transport.opendata.ch | Echtzeit-Verbindungen & Verspätungen | https://transport.opendata.ch |
        | SBB Open Data | Störungsmeldungen | https://data.sbb.ch |
        | Open-Meteo | Wetterdaten (historisch & live) | https://open-meteo.com |

        ---

        ### 🤖 ML-Modell
        - **Algorithmus:** Random Forest Classifier
        - **Features:** Wochentag, Uhrzeit, Temperatur, Niederschlag, Schneefall, Wind
        - **Output:** Risikolevel (Niedrig / Mittel / Hoch)

        ---

        ### 👥 Contribution-Matrix
        | Person | Aufgabe |
        |---|---|
        | Person 1 | TODO |
        | Person 2 | TODO |
        | Person 3 | TODO |
        | Person 4 | TODO |
        | Person 5 | TODO |
        """)


# ============================================================
# APP STARTEN
# ============================================================
if __name__ == "__main__":
    main()
