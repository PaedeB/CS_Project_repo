# ============================================================
# TEST-DATEI FÜR VS CODE (kein Streamlit nötig)
# ============================================================

import requests
import pandas as pd
import urllib.parse
from datetime import datetime, date


def st_cache_dummy(ttl=None):
    def decorator(func):
        return func
    return decorator

class st:
    cache_data = staticmethod(st_cache_dummy)

    @staticmethod
    def warning(msg):
        print(f"[WARNING] {msg}")


# ============================================================
# API: SBB IST-DATEN (data.sbb.ch)
# Echtzeit-Haltedaten → Verbindungen werden manuell verknüpft
# Dokumentation: https://data.sbb.ch/explore/dataset/ist-daten-sbb
# ============================================================

API_SBB_IST = "https://data.sbb.ch/api/explore/v2.1/catalog/datasets/ist-daten-sbb/records"

def parse_bool(value):
    """
    Wandelt API-Werte sicher in Boolean um.
    Das API liefert teilweise den String "false"/"true" statt Boolean.
    In Python ist bool("false") == True (nicht-leerer String!), deshalb
    muss explizit auf den String-Inhalt geprüft werden.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return False


def neuestes_datum_holen():
    """
    Holt das neueste verfügbare betriebstag aus dem Dataset.
    Das Dataset wird nicht täglich aktualisiert, deshalb kann
    'heute' im Dataset schlicht nicht existieren.
    """
    try:
        params = {"limit": 1, "order_by": "betriebstag DESC"}
        r = requests.get(API_SBB_IST, params=params, timeout=10)
        r.raise_for_status()
        records = r.json().get("results", [])
        if records:
            return records[0].get("betriebstag", "")
        return ""
    except requests.exceptions.RequestException:
        return ""


@st.cache_data(ttl=120)
def verbindungen_laden(von="Zürich HB", nach="Bern", anzahl=10):
    try:
        neuestes_datum = neuestes_datum_holen()
        print(f"[INFO] Verwende betriebstag: {neuestes_datum}")

        # --- [1] Nur IC/IR ab Startbahnhof laden ---
        # verkehrsmittel_text filtert S-Bahnen heraus → viel weniger Records nötig
        params_von = {
            "refine": [
                f"haltestellen_name:{von}",
                f"betriebstag:{neuestes_datum}",
            ],
            "where": 'verkehrsmittel_text in ("IC", "IR", "EC", "ICE", "RE")',
            "limit": 10000,
            "order_by": "abfahrtszeit ASC",
        }
        r_von = requests.get(API_SBB_IST, params=params_von, timeout=10)
        r_von.raise_for_status()
        records_von = r_von.json().get("results", [])

        if not records_von:
            st.warning(f"Keine IC/IR Records für '{von}' gefunden.")
            return pd.DataFrame()

        # --- [2] Fahrt-IDs extrahieren ---
        fahrt_ids = list({
            r["fahrt_bezeichner"]
            for r in records_von
            if r.get("fahrt_bezeichner")
        })

        # --- [3] Welche dieser Züge halten auch am Zielbahnhof? ---
        # where mit AND kombiniert Station + fahrt_bezeichner-Liste
        ids_filter  = ", ".join([f'"{fid}"' for fid in fahrt_ids[:50]])
        params_nach = {
            "refine": f"betriebstag:{neuestes_datum}",
            "where": f'haltestellen_name="{nach}" AND fahrt_bezeichner in ({ids_filter})',
            "limit": 10000,
        }
        r_nach = requests.get(API_SBB_IST, params=params_nach, timeout=10)
        r_nach.raise_for_status()
        records_nach = r_nach.json().get("results", [])

        if not records_nach:
            st.warning(f"Keine passenden Züge nach '{nach}' gefunden.")
            return pd.DataFrame()

        # --- [4] Zielbahnhof-Lookup aufbauen ---
        ziel_lookup = {
            r["fahrt_bezeichner"]: r
            for r in records_nach
            if r.get("fahrt_bezeichner")
        }

        # --- [5] Verbindungen zusammenführen + Richtung prüfen ---
        verbindungen = []
        for r in records_von:
            fid = r.get("fahrt_bezeichner")
            if fid not in ziel_lookup:
                continue

            ziel = ziel_lookup[fid]

            abfahrt_geplant = r.get("abfahrtszeit", "")
            ankunft_geplant = ziel.get("ankunftszeit", "")

            # Richtungsprüfung: Abfahrt muss VOR Ankunft liegen
            # Verhindert dass Züge Bern→Zürich ebenfalls erscheinen
            if abfahrt_geplant and ankunft_geplant:
                if abfahrt_geplant >= ankunft_geplant:
                    continue

            zugname        = r.get("linien_text", "–") or "–"
            verspaetung_ab = parse_bool(r.get("abfahrtsverspatung", False))
            verspaetung_an = parse_bool(ziel.get("ankunftsverspatung", False))

            verbindungen.append({
                "Zug":                zugname,
                "Abfahrt (geplant)":  abfahrt_geplant[:16] if abfahrt_geplant else "–",
                "Ankunft (geplant)":  ankunft_geplant[:16] if ankunft_geplant else "–",
                "Verspätung Abfahrt": "🔴 Ja" if verspaetung_ab else "🟢 Nein",
                "Verspätung Ankunft": "🔴 Ja" if verspaetung_an else "🟢 Nein",
            })

            if len(verbindungen) >= anzahl:
                break

        return pd.DataFrame(verbindungen)

    except requests.exceptions.RequestException as e:
        st.warning(f"Verbindungsfehler zur SBB IST-Daten-API: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=120)
def abfahrten_laden(station="Zürich HB", anzahl=20):
    try:
        neuestes_datum = neuestes_datum_holen()

        # Nur IC/IR für übersichtlicheres Abfahrtsbrett
        params = {
            "refine": [
                f"haltestellen_name:{station}",
                f"betriebstag:{neuestes_datum}",
            ],
            "where": 'verkehrsmittel_text in ("IC", "IR", "EC", "ICE", "RE")',
            "limit": anzahl,
            "order_by": "abfahrtszeit ASC",
        }
        response = requests.get(API_SBB_IST, params=params, timeout=10)
        response.raise_for_status()
        records = response.json().get("results", [])

        abfahrten = []
        for r in records:
            abfahrt     = r.get("abfahrtszeit", "")
            zugname     = r.get("linien_text", "–") or "–"
            verspaetung = parse_bool(r.get("abfahrtsverspatung", False))

            abfahrten.append({
                "Zug":       zugname,
                "Abfahrt":   abfahrt[11:16] if abfahrt else "–",
                "Verspätet": "🔴 Ja" if verspaetung else "🟢 Nein",
            })

        return pd.DataFrame(abfahrten)

    except requests.exceptions.RequestException as e:
        st.warning(f"Fehler beim Laden der Abfahrten: {e}")
        return pd.DataFrame()


# ============================================================
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

print()