# ============================================================
# KONFIGURATION
# Zentrale Einstellungen 
# enthält alle wichtigen Parameter, auf die andere Module zugreifen können.
# ============================================================

# ------------------------------------------------------------
# PROJEKT / ALLGEMEINE EINSTELLUNGEN
# ------------------------------------------------------------
PROJEKT_NAME = "SBB Reisezuverlässigkeits-Analyzer & Prediktor"
STANDARD_STRECKE_NAME = "St.Gallen → Zürich HB "

# ------------------------------------------------------------
# ORTE / STATIONEN
# Für relevante Bahnhöfe werden die SBB-Stations-ID und die
# Koordinaten gespeichert.
# Die station_ID wird für Transportdaten gebraucht, die Koordinaten
# vor allem für die Wetterdaten 
# ------------------------------------------------------------
ORTE = {
    "St. Gallen": {
        "station_id": "8506302",
        "lat": 47.4245,
        "lon": 9.3696
    },
    "Winterthur": {
        "station_id": "8506000",
        "lat": 47.5003,
        "lon": 8.7241
    },
    "Zürich Flughafen": {
        "station_id": "8503016",
        "lat": 47.4506,
        "lon": 8.5610
    },
    "Zürich HB": {
        "station_id": "8503000",
        "lat": 47.3782,
        "lon": 8.5402
    }
}

# ------------------------------------------------------------
# STRECKE
# Definiert die Standardstrecke der Analyse
# Zwischenhalte sind optional und können später erweitert werden.
# ------------------------------------------------------------
STRECKE = {
    "name": "St. Gallen → Zürich HB",
    "von": "St. Gallen",
    "nach": "Zürich HB",
    "von_id": ORTE["St. Gallen"]["station_id"],
    "nach_id": ORTE["Zürich HB"]["station_id"],
    "zwischenhalte": ["Winterthur", "Zürich Flughafen"]
}
# ------------------------------------------------------------
# WETTERBEZUG
# Legt fest, für welchen Ort standardmässig Wetterdaten geladen werden.
# ------------------------------------------------------------
WETTER_STANDARD_ORT = "St. Gallen"
WETTER_LAT = ORTE[WETTER_STANDARD_ORT]["lat"]
WETTER_LON = ORTE[WETTER_STANDARD_ORT]["lon"]

# ------------------------------------------------------------
# ZEITPARAMETER
# Zeitraum für historische Daten und geplante Reisen.
# ------------------------------------------------------------
HIST_START = "2024-01-01"
HIST_ENDE = "2024-12-31"    #anpassen!!

# ------------------------------------------------------------
# API-ENDPUNKTE 
# URLs der verwendeten APIs. 
# ------------------------------------------------------------
API_TRANSPORT   = "https://transport.opendata.ch/v1"
API_STOERUNGEN  = "https://www.sbb.ch/content/dam/internet/sbb/de/meta/footer/baustellen.xml"
API_WETTER      = "https://api.open-meteo.com/v1/forecast"
API_WETTER_HIST = "https://archive-api.open-meteo.com/v1/archive"

# ------------------------------------------------------------
# APP- UND ABFRAGEPARAMETER
# zentrale Standardwerte für Datenabfragen 
# ------------------------------------------------------------
MAX_VERBINDUNGEN = 10 #Anzahl Verbindungen pro API-Abfrage

# ------------------------------------------------------------
# OPTIONALE CACHE-ZEITEN
# Diese Werte können später verwendet werden, um festzulegen,
# wie lange API-Daten zwischengespeichert werden.
# Dadurch werden unnötige API-Aufrufe reduziert und die App läuft schneller. 
# ------------------------------------------------------------
CACHE_TRANSPORT_SEK = 120
CACHE_STOERUNGEN_SEK = 300
CACHE_WETTER_SEK = 3600
CACHE_WETTER_HIST_SEK = 86400
