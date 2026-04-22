
import requests
import pandas as pd
import streamlit as st

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
