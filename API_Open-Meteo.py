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
