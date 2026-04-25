# ============================================================
# DATENBANK SETUPa
# Speichert historische Abfragen lokal für ML-Training
# ============================================================
import streamlit as st
import pandas as pd


from API_Open_Meteo import get_historical #importierung der historischen Wetterdaten
from API_SBB_IST import verbindungen_laden, abfahrten_laden #importierung der aktuellen Verbindungen
from API_SBB_Störungsmeldung import stoerungen_laden_api #importierung der aktuellen Störungen
from konfiguration import ORTE, STRECKE

# ── 1. Daten laden ─────────────────────────────────────────
#Einschränkung auf 10000 aufgrund der Datenmenge
sbb = verbindungen_laden(von="St. Gallen", nach="Zürich HB", anzahl=10000)
sbb = pd.concat([verbindungen_laden(von="Zürich HB", nach="St. Gallen", anzahl=10000), sbb])

abfrage_stardat = str(sbb["Abfahrt (geplant)"].min())[:10]
abfrage_enddat = str(sbb["Abfahrt (geplant)"].max())[:10]

wetter = get_historical(47.3782, 8.5403, abfrage_stardat, abfrage_enddat)
#stoerungen = stoerungen_laden_api()

print(wetter)
print(abfrage_stardat)
print(abfrage_enddat)

# ── 2. Unique Key erstellen ─────────────────────────────────
sbb["Unique Zug & Abfahrt"] = sbb["Zug"] + "_" + sbb["Abfahrt (geplant)"]

# ── 3. Datum-Spalte aus Abfahrt extrahieren (für Merge) ────
sbb["datum"] = pd.to_datetime(sbb["Abfahrt (geplant)"]).dt.date
wetter["datum"] = pd.to_datetime(wetter["date"]).dt.date

# ── 4. Verspätung als 0/1 umwandeln (für ML) ───────────────
sbb["verspätet"] = sbb["Verspätung Abfahrt"].apply(
    lambda x: 1 if "Ja" in x else 0
)

# ── 5. Störung vorhanden? (für ML) ─────────────────────────
#stoerungen["datum"] = pd.to_datetime(stoerungen["Datum"]).dt.date
#stoerungen["stoerung_aktiv"] = 1

# ── 6. Alles zusammenführen ─────────────────────────────────
df = pd.merge(sbb, wetter, on="datum", how="left")
#df = pd.merge(df, stoerungen[["datum", "stoerung_aktiv"]], on="datum", how="left")
#df["stoerung_aktiv"] = df["stoerung_aktiv"].fillna(0).astype(int)

# ── 7. Nur relevante Spalten behalten ──────────────────────
df_ml = df[[
    "Unique Zug & Abfahrt",   # Key
    "datum",                   # Datum
    "Zug",                     # Zuglinie
    "temperature_2m_max",      # Temperatur
    "precipitation_sum",       # Niederschlag
    "snowfall_sum",            # Schneefall
    "windspeed_10m_max",       # Wind
    #"stoerung_aktiv",          # Störung ja/nein
    "verspätet"                # ZIEL für ML (0=pünktlich, 1=verspätet)
]]

# ── Störungen: Pro Datum nur 1 Zeile (nicht mehrere) ───────
#stoerungen["datum"] = pd.to_datetime(stoerungen["Datum"]).dt.date
#stoerungen_grouped = stoerungen.groupby("datum")["stoerung_aktiv"].max().reset_index()

# ── Merge mit deduplizierten Störungen ──────────────────────
df = pd.merge(sbb, wetter, on="datum", how="left")
#df = pd.merge(df, stoerungen_grouped, on="datum", how="left")
#df["stoerung_aktiv"] = df["stoerung_aktiv"].fillna(0).astype(int)

df = pd.merge(sbb, wetter, on="datum", how="left")
#df = pd.merge(df, stoerungen_grouped, on="datum", how="left")
#df["stoerung_aktiv"] = df["stoerung_aktiv"].fillna(0).astype(int)

# Duplikate entfernen
df = df.drop_duplicates(subset=["Unique Zug & Abfahrt"])

# Nur EC-Züge behalten
df_ml = df[df["Zug"] == "EC"].reset_index(drop=True)

# Nur relevante Spalten für ML behalten
df_ml = df_ml[[
    "Unique Zug & Abfahrt",
    "datum",
    "Abfahrt (geplant)",
    "temperature_2m_max",
    "precipitation_sum",
    "snowfall_sum",
    "windspeed_10m_max",
    #"stoerung_aktiv",
    "verspätet"            # Ziel für ML
]].reset_index(drop=True)

print(df_ml)
print(f"\nForm: {df_ml.shape}")  # (Zeilen, Spalten)