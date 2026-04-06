# ============================================================
# KONFIGURATION
# Zentrale Einstellungen – hier anpassen falls nötig
# ============================================================

# Analysierte Strecke 
STRECKE = {
    "name": "Bern → Zürich HB",
    "von": "Bern",
    "nach": "Zürich HB",
    "von_id": "8507000",   # SBB Stations-ID Bern
    "nach_id": "8503000",  # SBB Stations-ID Zürich HB
}

# API Endpoints
API_TRANSPORT   = "https://transport.opendata.ch/v1"
API_STOERUNGEN  = "https://www.sbb.ch/content/dam/internet/sbb/de/meta/footer/baustellen.xml"
API_WETTER      = "https://api.open-meteo.com/v1/forecast"
API_WETTER_HIST = "https://archive-api.open-meteo.com/v1/archive"

# Koordinaten Bern (für Wetter-API)
BERN_LAT = 46.9480
BERN_LON = 7.4474

# Zeitraum für historische Daten
HIST_START = "2024-01-01" #muss noch angepasst werden
HIST_ENDE  = "2024-12-31"

# Anzahl Verbindungen die abgerufen werden
MAX_VERBINDUNGEN = 10
