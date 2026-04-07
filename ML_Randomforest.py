# ============================================================
# MASCHINELLES LERNEN
# Random Forest Classifier für Verspätungsrisiko
# ============================================================

def trainings_daten_generieren(n=500):
    """
    Generiert synthetische Trainingsdaten bis echte historische
    Daten aus der Datenbank verfügbar sind.

    WICHTIG: Diese Funktion durch echte Daten ersetzen sobald
    genug Datenpunkte in der DB gesammelt wurden (ca. 2-3 Wochen)!

    Muster (basierend auf bekannten SBB-Mustern):
    - Rushhour (7-9h, 17-19h) → mehr Verspätungen
    - Schneefall > 2cm        → hohes Risiko
    - Temperatur < 0°C        → erhöhtes Risiko
    - Wochenende              → weniger Verspätungen
    """
    np.random.seed(42)
    n_samples = n

    wochentag   = np.random.randint(0, 7, n_samples)
    stunde      = np.random.randint(5, 23, n_samples)
    temperatur  = np.random.uniform(-10, 35, n_samples)
    niederschlag = np.random.uniform(0, 20, n_samples)
    schneefall  = np.random.uniform(0, 15, n_samples)
    wind        = np.random.uniform(0, 80, n_samples)

    # Risiko-Logik für synthetische Labels
    risiko = []
    for i in range(n_samples):
        score = 0
        # Rushhour
        if stunde[i] in [7, 8, 9, 17, 18, 19]:    score += 2
        # Schnee
        if schneefall[i] > 5:                       score += 3
        elif schneefall[i] > 2:                     score += 1
        # Kälte
        if temperatur[i] < -5:                      score += 2
        elif temperatur[i] < 0:                     score += 1
        # Starker Wind
        if wind[i] > 60:                            score += 2
        elif wind[i] > 40:                          score += 1
        # Starker Regen
        if niederschlag[i] > 10:                    score += 1
        # Wochenende (weniger Betrieb → weniger Verspätungen)
        if wochentag[i] in [5, 6]:                  score -= 1

        if score <= 1:   risiko.append(0)   # Niedrig
        elif score <= 3: risiko.append(1)   # Mittel
        else:            risiko.append(2)   # Hoch

    df = pd.DataFrame({
        "wochentag":        wochentag,
        "stunde":           stunde,
        "temperatur":       temperatur,
        "niederschlag":     niederschlag,
        "schneefall":       schneefall,
        "windgeschwindigkeit": wind,
        "risiko":           risiko
    })
    return df

@st.cache_resource  # Modell wird nur einmal trainiert und gecacht
def ml_modell_trainieren():
    """
    Trainiert den Random Forest Classifier.
    Versucht zuerst echte Daten aus der DB zu laden.
    Falls zu wenig Daten vorhanden, werden synthetische verwendet.
    Gibt das trainierte Modell und die Feature-Namen zurück.
    """
    # Echte Daten versuchen zu laden
    echte_daten = daten_laden()

    if len(echte_daten) >= 100:
        # Genug echte Daten vorhanden!
        df = echte_daten.copy()
        # Risiko-Label aus Verspätungsminuten ableiten
        df["risiko"] = df["verspaetung_min"].apply(
            lambda x: 0 if x <= 3 else (1 if x <= 10 else 2)
        )
    else:
        # Fallback: Synthetische Daten
        df = trainings_daten_generieren(500)

    # Features definieren (was das Modell als Input bekommt)
    features = ["wochentag", "stunde", "temperatur",
                "niederschlag", "schneefall", "windgeschwindigkeit"]

    X = df[features]
    y = df["risiko"]

    # Modell trainieren
    modell = RandomForestClassifier(
        n_estimators=100,    # 100 Entscheidungsbäume
        max_depth=8,         # Tiefe begrenzen (verhindert Overfitting)
        random_state=42
    )
    modell.fit(X, y)

    return modell, features

def risiko_vorhersagen(modell, features, wochentag, stunde,
                       temperatur, niederschlag, schneefall, wind):
    """
    Sagt das Verspätungsrisiko für gegebene Bedingungen voraus.
    Gibt zurück: Risikolevel (0/1/2), Label, Wahrscheinlichkeiten,
                 und Feature Importances.
    """
    # Input als DataFrame (Modell erwartet dieses Format)
    X = pd.DataFrame([[wochentag, stunde, temperatur,
                        niederschlag, schneefall, wind]],
                     columns=features)

    # Vorhersage
    risiko_num    = modell.predict(X)[0]
    wahrscheinl   = modell.predict_proba(X)[0]

    # Feature Importance (welche Faktoren sind am wichtigsten?)
    importances = dict(zip(features, modell.feature_importances_))

    # Numerisches Label in Text umwandeln
    labels = {0: "🟢 Niedrig", 1: "🟡 Mittel", 2: "🔴 Hoch"}
    return risiko_num, labels[risiko_num], wahrscheinl, importances
