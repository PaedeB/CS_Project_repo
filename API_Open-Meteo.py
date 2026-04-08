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

ABDI START

import requests
import pandas as pd

# ── Forecast (next 7 days) ─────────────────────────────────────────────────────

def get_forecast(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "precipitation_sum", "snowfall_sum", "windspeed_10m_max"],
        "timezone": "Europe/Zurich",
        "forecast_days": 7,
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["daily"])
    df.rename(columns={"time": "date"}, inplace=True)
    return df


# ── Historical (past dates) ────────────────────────────────────────────────────

def get_historical(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,   # "YYYY-MM-DD"
        "end_date": end_date,       # "YYYY-MM-DD"
        "daily": ["temperature_2m_max", "precipitation_sum", "snowfall_sum", "windspeed_10m_max"],
        "timezone": "Europe/Zurich",
    }
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["daily"])
    df.rename(columns={"time": "date"}, inplace=True)
    return df


# ── Example usage ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    lat, lon = 47.3782, 8.5403  # Zürich HB

    print("--- Forecast ---")
    print(get_forecast(lat, lon))

    print("--- Historical ---")
    print(get_historical(lat, lon, "2023-01-01", "2023-12-31"))