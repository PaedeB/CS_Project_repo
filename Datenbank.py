# ============================================================
# DATENBANK SETUPa
# Speichert historische Abfragen lokal für ML-Training
# ============================================================
import streamlit as st
import pandas as pd

from API_Open_Meteo import get_historical #importierung der historischen Wetterdaten
from API_SBB_IST import verbindungen_laden, abfahrten_laden #importierung der aktuellen Verbindungen
from API_SBB_Störungsmeldung import stoerungen_laden_api #importierung der aktuellen Störungen


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

#Testspace
print(stoerungen_laden_api())
