
import streamlit as st #diese und folgende 3 Zeilen braucht es nur einmal im gesamten Code
import requests
import pandas as pd
from datetime import datetime

# ============================================================
# API: TRANSPORT.OPENDATA.CH
# Lädt aktuelle Verbindungen und Verspätungen
# ============================================================

API_TRANSPORT = "https://transport.opendata.ch/v1"

@st.cache_data(ttl=120)  # 2 Minuten Cache (Echtzeit-Daten)
def verbindungen_laden(von, nach, anzahl=10): #Ruft aktuelle Zugverbindungen von transport.opendata.ch ab. Gibt einen DataFrame mit Verbindungen inkl. Verspätungen zurück. Dokumentation: https://transport.opendata.ch/docs.html
    try: # API-Parameter zusammenstellen 
        params = {
            "from": von, #Start- und Zielbahnhof als Klartext (z.B. Zürich HB)
            "to": nach,
            "limit": anzahl, #maximale Anzahl zurückgegebener Verbindungen 
            "fields[]": ["connections/from", "connections/to", # schränkt die API-Antwort auf benötigte Felder ein & spart somit Bandbreite 
                         "connections/duration", "connections/products"]
        }
        response = requests.get(f"{API_TRANSPORT}/connections", params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        verbindungen = [] 
        for conn in data.get("connections", []):
            abfahrt_geplant = conn.get("from", {}).get("departure", "") # Hole "from" (oder {} wenn nicht da), daraus dann "departure" (oder "")
            abfahrt_ist     = conn.get("from", {}).get("prognosis", {}).get("departure", "") # Gleiche Logik wie oben, aber hier wird zusätzlich "prognosis" abgefragt, um die Echtzeit-Abfahrtszeit zu bekommen
            ankunft_geplant = conn.get("to", {}).get("arrival", "")
            ankunft_ist     = conn.get("to", {}).get("prognosis", {}).get("arrival", "")

            # Verspätung berechnen (falls Echtzeit-Daten vorhanden)
            verspaetung = 0
            if abfahrt_geplant and abfahrt_ist:
                try:
                    geplant_dt = datetime.fromisoformat(abfahrt_geplant[:19]) #Nimmt die ersten 19 Zeichen der Zeit (bsp. 2026-04-08T12:02:00); es werden also die Zeitzoneninformationen abgeschnitten, da diese manchmal zu Problemen führen können; die Sekunden sind hier jedoch drin.
                    ist_dt     = datetime.fromisoformat(abfahrt_ist[:19])
                    verspaetung = max(0, (ist_dt - geplant_dt).total_seconds // 60)
                except:
                    verspaetung = 0

            verbindungen.append({
                "Abfahrt (geplant)": abfahrt_geplant[:16] if abfahrt_geplant else "-", #Nimmt die ersten 16 Zeichen der Zeit (bsp. 2026-04-08T12:02); es werden also die Sekunden abgeschnitten
                "Abfahrt (ist)":     abfahrt_ist[:16]     if abfahrt_ist     else "-",
                "Ankunft (geplant)": ankunft_geplant[:16] if ankunft_geplant else "-",
                "Verspätung (min)":  verspaetung,
                "Status": "🔴 Verspätet" if verspaetung > 3 else "🟢 Pünktlich" #wurde bewusst grösser als 3 ausgewählt, da es bei der SBB auch erst dann als Verspätung angezeigt wird
            })

        return pd.DataFrame(verbindungen)

    except requests.exceptions.RequestException as e: #Fehlerbehandlung, schützt vor Internetproblemen oder API-Ausfällen
        st.warning(f"Verbindungsfehler zur Transport-API: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=120) # 2 Minuten Cache (Echtzeit-Daten)
def abfahrten_laden(station, anzahl=20): # Lädt aktuelle Abfahrten an einem Bahnhof, nützlich für das Echtzeit-Dashboard
    try: 
        params = {"station": station, "limit": anzahl} # API-Parameter: Bahnhof-ID und Anzahl der Abfahrten
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
                "Zug":             journey.get("number", "-"),
                "Richtung":        journey.get("to", "-"),
                "Abfahrt":         abfahrt[11:16] if abfahrt else "-",
                "Verspätung (min)": verspaetung,
                "Status":          "🔴 Verspätet" if verspaetung > 3 else "🟢 Pünktlich"
            })

        return pd.DataFrame(abfahrten)

    except requests.exceptions.RequestException as e:
        st.warning(f"Fehler beim Laden der Abfahrten: {e}")
        return pd.DataFrame()
    
if __name__ == "__main__": #diese und die nächsten 2 Zeilen, sind nur drin um die funktionalität zu testen; müssen bei der Zusammenführung des Codes gelöscht werden!
    df = verbindungen_laden("Zürich", "Bern", 5)
    print(df)

