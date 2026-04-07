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
