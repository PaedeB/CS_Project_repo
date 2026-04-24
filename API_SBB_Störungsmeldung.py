# ========================================================
# API: SBB STÖRUNGSMELDUNGEN
# Lädt aktuelle Störungen über die offizielle SBB Open Data API
# Fokus: Nur EC-Züge zwischen St. Gallen und Zürich
# ========================================================

# Bibliotheken importieren
import requests   # für HTTP-Anfragen an die SBB API
import pandas as pd  # für die Verarbeitung der Daten als Tabelle
import streamlit as st  # für die Web-App und den Cache
import re  # für das Durchsuchen von Text (reguläre Ausdrücke)

# API URL der SBB Open Data Plattform für Störungsmeldungen
API_STOERUNGEN = "https://data.sbb.ch/api/explore/v2.1/catalog/datasets/rail-traffic-information/records"

def linien_extrahieren(description):
    """
    Extrahiert Zuglinien aus der Beschreibung.
    Beispiel: 'Lines IC1, IR15, RE33 are affected.' -> 'IC1, IR15, RE33'
    """
    # Falls keine Beschreibung vorhanden, leeren String zurückgeben
    if not description:
        return ""
    
    # Sucht nach dem Muster "Lines X, Y and Z are affected" im Text
    # r'Lines?\s+...' erlaubt sowohl "Line" als auch "Lines"
    match = re.search(r'Lines?\s+([\w\d,\s]+?)\s+(are|is)\s+affected', description)
    
    if match:
        # Extrahiert nur den Teil mit den Liniennamen (Gruppe 1)
        linien = match.group(1)
        # Ersetzt " and " durch ", " für einheitliches Format
        linien = linien.replace(" and ", ", ")
        return linien.strip()
    
    # Kein Muster gefunden → leeren String zurückgeben
    return ""

# Cache: Ergebnis wird 5 Minuten gespeichert um API-Aufrufe zu reduzieren
@st.cache_data(ttl=300)
def stoerungen_laden_api():
    """
    Lädt aktuelle SBB-Störungsmeldungen via Open Data API.
    Gibt einen pandas DataFrame zurück mit Datum, Störung und Beschreibung.
    Gefiltert auf EC-Züge.
    """
    try:
        # Parameter für die API-Anfrage:
        # limit=100 → maximal 100 Störungen laden
        # order_by → neueste Störungen zuerst
        params = {
            "limit": 100,
            "order_by": "startdatetime desc"
        }
        
        # HTTP GET-Anfrage an die SBB API senden
        response = requests.get(API_STOERUNGEN, params=params, timeout=10)
        
        # Fehler auslösen falls die API nicht erreichbar ist (z.B. 404, 500)
        response.raise_for_status()

        # Antwort der API von JSON in Python-Dictionary umwandeln
        data = response.json()
        
        # Leere Liste für die verarbeiteten Störungen
        stoerungen = []

        # Jede Störung aus der API-Antwort verarbeiten
        for item in data["results"]:
            stoerungen.append({
                "Datum":        item.get("startdatetime", ""),  # Startzeit der Störung
                "Störung":      item.get("title", ""),          # Titel/Name der Störung
                "Beschreibung": item.get("description", ""),    # Detailbeschreibung
                "Typ":          item.get("type", ""),           # Typ (2=begrenzt, 5=unterbrochen)
                "Ende":         item.get("enddatetime", ""),    # Geplantes Ende der Störung
                "Linien":       linien_extrahieren(item.get("description", ""))  # Betroffene Zuglinien
            })

        # Liste in pandas DataFrame umwandeln für einfache Weiterverarbeitung
        df = pd.DataFrame(stoerungen)
        
        # Nur EC-Störungen behalten (Fokus auf EC St. Gallen - Zürich)
        # na=False verhindert Fehler bei leeren Linien-Feldern
        df_ec = df[df["Linien"].str.contains("EC", na=False)]
        
        return df_ec

    except Exception as e:
        # Falls API nicht erreichbar: Warnung anzeigen und leere Tabelle zurückgeben
        # So stürzt die App nicht ab sondern zeigt eine hilfreiche Meldung
        st.warning(f"API nicht erreichbar: {e}")
        return pd.DataFrame(columns=["Datum", "Störung", "Beschreibung", "Typ", "Ende", "Linien"])