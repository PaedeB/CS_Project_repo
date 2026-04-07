# ============================================================
# API: SBB STÖRUNGSMELDUNGEN
# Lädt aktuelle Störungen als RSS-Feed
# ============================================================

@st.cache_data(ttl=300)  # 5 Minuten Cache
def stoerungen_laden_api():
    """
    Lädt aktuelle SBB-Störungsmeldungen.

    TODO für die Gruppe: Den korrekten RSS-Feed-Link der SBB einfügen.
    Alternativ kann auch die offizielle SBB-Störungsseite gescrapt werden.
    Überprüft auf: https://www.sbb.ch/de/kaufen/pages/fahrplan/aktuell.xhtml
    """
    try:
        response = requests.get(API_STOERUNGEN, timeout=10)
        response.raise_for_status()

        # XML parsen
        root = ET.fromstring(response.content)
        stoerungen = []
        for item in root.findall(".//item"):
            titel = item.findtext("title", "")
            beschr = item.findtext("description", "")
            datum  = item.findtext("pubDate", "")
            stoerungen.append({
                "Datum":        datum,
                "Störung":      titel,
                "Beschreibung": beschr
            })
        return pd.DataFrame(stoerungen)

    except Exception as e:
        # Fallback: Leere Liste mit Hinweis
        # (API-Link muss noch verifiziert werden)
        return pd.DataFrame(columns=["Datum", "Störung", "Beschreibung"])
