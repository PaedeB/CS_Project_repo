# ========================================================
# API: SBB STÖRUNGSMELDUNGEN
# Lädt aktuelle Störungen über die offizielle SBB Open Data API
# ========================================================

import requests
import pandas as pd
import streamlit as st

# API URL
API_STOERUNGEN = "https://data.sbb.ch/api/explore/v2.1/catalog/datasets/rail-traffic-information/records"

@st.cache_data(ttl=300)  # 5 Minuten Cache
def stoerungen_laden_api():
    """
    Lädt aktuelle SBB-Störungsmeldungen via Open Data API.
    Gibt einen pandas DataFrame zurück mit Datum, Störung und Beschreibung.
    """
    try:
        params = {
            "limit": 100,
            "order_by": "startdatetime desc"
        }
        response = requests.get(API_STOERUNGEN, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        stoerungen = []

        for item in data["results"]:
            stoerungen.append({
                "Datum":        item.get("startdatetime", ""),
                "Störung":      item.get("title", ""),
                "Beschreibung": item.get("description", ""),
                "Typ":          item.get("type", ""),
                "Ende":         item.get("enddatetime", "")
            })

        return pd.DataFrame(stoerungen)

    except Exception as e:
        # Fallback: leere Liste mit Hinweis
        st.warning(f"API nicht erreichbar: {e}")
        return pd.DataFrame(columns=["Datum", "Störung", "Beschreibung", "Typ", "Ende"])

if __name__ == "__main__":
    df = stoerungen_laden_api()
[   print(df.head())
 #Lantwin Task: 1. Code Problem fixen 2. Zugname separat in Tabelle auflisten lassen aus der Description