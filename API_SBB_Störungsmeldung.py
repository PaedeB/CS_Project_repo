# ========================================================
# API: SBB STÖRUNGSMELDUNGEN
# Lädt aktuelle Störungen über die offizielle SBB Open Data API
# ========================================================

import requests
import pandas as pd
import streamlit as st
import re

# API URL
API_STOERUNGEN = "https://data.sbb.ch/api/explore/v2.1/catalog/datasets/rail-traffic-information/records"

def linien_extrahieren(description):
    """
    Extrahiert Zuglinien aus der Beschreibung.
    Beispiel: 'Lines IC1, IR15, RE33 are affected.' -> 'IC1, IR15, RE33'
    """
    if not description:
        return ""
    match = re.search(r'Lines?\s+([\w\d,\s]+?)\s+(are|is)\s+affected', description)
    if match:
        linien = match.group(1)
        linien = linien.replace(" and ", ", ")
        return linien.strip()
    return ""

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
                "Ende":         item.get("enddatetime", ""),
                "Linien":       linien_extrahieren(item.get("description", ""))
            })

        df = pd.DataFrame(stoerungen)
        df_ec = df[df["Linien"].str.contains("EC", na=False)]
        return df_ec

    except Exception as e:
        st.warning(f"API nicht erreichbar: {e}")
        return pd.DataFrame(columns=["Datum", "Störung", "Beschreibung", "Typ", "Ende", "Linien"])
