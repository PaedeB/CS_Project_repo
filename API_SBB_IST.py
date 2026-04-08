import streamlit as st #diese und folgende 3 Zeilen braucht es nur einmal im gesamten Code
import requests
import pandas as pd
from datetime import datetime

   # ============================================================
# TEST-DATEI FÜR VS CODE (kein Streamlit nötig)
# ============================================================

# Streamlit-Ersatz: Decorator und Warning einfach deaktivieren
def st_cache_dummy(ttl=None):
    def decorator(func):
        return func
    return decorator

class st:
    cache_data = staticmethod(st_cache_dummy)

    @staticmethod
    def warning(msg):
        print(f"[WARNING] {msg}") # bis hier später alles noch rauslöschen, da nur für code testen


# ============================================================
# API: SBB IST-DATEN (data.sbb.ch)
# Echtzeit-Haltedaten → Verbindungen werden manuell verknüpft
# Dokumentation: https://data.sbb.ch/explore/dataset/ist-daten-sbb
# ============================================================

API_SBB_IST = "https://data.sbb.ch/api/explore/v2.1/catalog/datasets/ist-daten-sbb/records"


@st.cache_data(ttl=120)  # 2 Minuten Cache
def verbindungen_laden(von="Zürich HB", nach="Bern", anzahl=10):
    """
    Rekonstruiert Verbindungen zwischen zwei Bahnhöfen aus den SBB IST-Daten.
    Da die API haltepunktbasiert ist, sind zwei Abfragen nötig:
      1. Alle Abfahrten ab Startbahnhof laden
      2. Prüfen, welche Züge davon auch am Zielbahnhof halten
    Die Züge werden über 'fahrt_bezeichner' verknüpft.
    """
    try:
        # --- [1] Schritt 1: Abfahrten ab Startbahnhof laden ---
        # "where": filtert nach Haltepunkt und schliesst ausgefallene Züge aus
        # "order_by": sortiert nach geplanter Abfahrtszeit aufsteigend
        params_von = {
            "where": f'haltestellen_name="{von}" AND faellt_aus_tf=false',
            "limit": 50,  # mehr laden als nötig, da nicht alle nach Bern fahren
            "order_by": "abfahrtszeit ASC",
        }
        r_von = requests.get(API_SBB_IST, params=params_von, timeout=10)
        r_von.raise_for_status()
        records_von = r_von.json().get("results", [])

        if not records_von:
            return pd.DataFrame()

        # --- [2] Fahrt-IDs (eindeutige Zug-Identifikation) extrahieren ---
        # "fahrt_bezeichner" ist der Schlüssel, über den wir beide Abfragen verknüpfen
        fahrt_ids = list({
            r["fahrt_bezeichner"]
            for r in records_von
            if r.get("fahrt_bezeichner")
        })

        # --- [3] Schritt 2: Welche dieser Züge halten auch in Bern? ---
        # Opendatasoft-Syntax: fahrt_bezeichner in ("id1", "id2", ...)
        ids_filter = ", ".join([f'"{fid}"' for fid in fahrt_ids[:30]])
        params_nach = {
            "where": f'haltestellen_name="{nach}" AND fahrt_bezeichner in ({ids_filter})',
            "limit": 50,
        }
        r_nach = requests.get(API_SBB_IST, params=params_nach, timeout=10)
        r_nach.raise_for_status()
        records_nach = r_nach.json().get("results", [])

        # --- [4] Bern-Datensätze als Lookup-Dictionary aufbauen ---
        # Schlüssel: fahrt_bezeichner → schneller Zugriff beim Zusammenführen
        bern_lookup = {
            r["fahrt_bezeichner"]: r
            for r in records_nach
            if r.get("fahrt_bezeichner")
        }

        # --- [5] Verbindungen zusammenführen und Verspätung berechnen ---
        verbindungen = []
        for r in records_von:
            fid = r.get("fahrt_bezeichner")

            # Nur Züge behalten, die auch in Bern halten
            if fid not in bern_lookup:
                continue

            bern = bern_lookup[fid]

            abfahrt_geplant = r.get("abfahrtszeit", "")
            abfahrt_ist     = r.get("ab_prognose", "")
            ankunft_geplant = bern.get("ankunftszeit", "")
            ankunft_ist     = bern.get("an_prognose", "")

            # Zugname aus produkt_id + linien_text zusammensetzen → z. B. "IC5", "IR13"
            produkt  = r.get("produkt_id", "")
            linie    = r.get("linien_text", "")
            zugname  = f"{produkt}{linie}".strip() if (produkt or linie) else "–"

            # Verspätung in Minuten (total_seconds() für korrekte Berechnung)
            verspaetung = 0
            if abfahrt_geplant and abfahrt_ist:
                try:
                    geplant_dt  = datetime.fromisoformat(abfahrt_geplant[:19])
                    ist_dt      = datetime.fromisoformat(abfahrt_ist[:19])
                    verspaetung = max(0, int((ist_dt - geplant_dt).total_seconds()) // 60)
                except (ValueError, TypeError):
                    verspaetung = 0

            verbindungen.append({
                "Zug":               zugname,
                "Abfahrt (geplant)": abfahrt_geplant[:16] if abfahrt_geplant else "–",
                "Abfahrt (ist)":     abfahrt_ist[:16]     if abfahrt_ist     else "–",
                "Ankunft (geplant)": ankunft_geplant[:16] if ankunft_geplant else "–",
                "Ankunft (ist)":     ankunft_ist[:16]     if ankunft_ist     else "–",
                "Verspätung (min)":  verspaetung,
                "Ausgefallen":       "❌ Ja" if r.get("faellt_aus_tf") else "✅ Nein",
                "Status":            "🔴 Verspätet" if verspaetung > 3 else "🟢 Pünktlich"
            })

            if len(verbindungen) >= anzahl:
                break

        return pd.DataFrame(verbindungen)

    except requests.exceptions.RequestException as e:
        st.warning(f"Verbindungsfehler zur SBB IST-Daten-API: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=120)
def abfahrten_laden(station="Zürich HB", anzahl=20):
    """
    Lädt aktuelle Abfahrten an einem Bahnhof aus den SBB IST-Daten.
    Hier genügt eine einzige API-Abfrage, da nur ein Haltepunkt relevant ist.
    """
    try:
        # --- [6] Einfache Abfrage: alle Abfahrten am gewünschten Bahnhof ---
        params = {
            "where": f'haltestellen_name="{station}"',
            "limit": anzahl,
            "order_by": "abfahrtszeit ASC",
        }
        response = requests.get(API_SBB_IST, params=params, timeout=10)
        response.raise_for_status()
        records = response.json().get("results", [])

        abfahrten = []
        for r in records:
            abfahrt  = r.get("abfahrtszeit", "")
            prognose = r.get("ab_prognose", "")

            # Zugname aus produkt_id + linien_text zusammensetzen → z. B. "IC5", "IR13"
            produkt  = r.get("produkt_id", "")
            linie    = r.get("linien_text", "")
            zugname  = f"{produkt}{linie}".strip() if (produkt or linie) else "–"

            verspaetung = 0
            if abfahrt and prognose:
                try:
                    a_dt        = datetime.fromisoformat(abfahrt[:19])
                    p_dt        = datetime.fromisoformat(prognose[:19])
                    verspaetung = max(0, int((p_dt - a_dt).total_seconds()) // 60)
                except (ValueError, TypeError):
                    pass

            # abfahrt[11:16] → schneidet "2025-04-08T14:32:00+02:00" auf "14:32"
            abfahrten.append({
                "Zug":              zugname,
                "Abfahrt":          abfahrt[11:16] if abfahrt else "–",
                "Verspätung (min)": verspaetung,
                "Ausgefallen":      "❌ Ja" if r.get("faellt_aus_tf") else "✅ Nein",
                "Status":           "🔴 Verspätet" if verspaetung > 3 else "🟢 Pünktlich"
            })

        return pd.DataFrame(abfahrten)

    except requests.exceptions.RequestException as e:
        st.warning(f"Fehler beim Laden der Abfahrten: {e}")
        return pd.DataFrame()
    

# ============================================================ # wird später gelöscht, da nur für code testen
# TEST-AUFRUFE
# ============================================================
if __name__ == "__main__":

    print("\n--- TEST: verbindungen_laden ---")
    df_verbindungen = verbindungen_laden(von="Zürich HB", nach="Bern", anzahl=5)
    if df_verbindungen.empty:
        print("Keine Verbindungen gefunden.")
    else:
        print(df_verbindungen.to_string(index=False))

    print("\n--- TEST: abfahrten_laden ---")
    df_abfahrten = abfahrten_laden(station="Zürich HB", anzahl=5)
    if df_abfahrten.empty:
        print("Keine Abfahrten gefunden.")
    else:
        print(df_abfahrten.to_string(index=False))