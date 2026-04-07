# ============================================================
# HILFSFUNKTIONEN
# ============================================================

def wochentag_name(nr):
    """Wandelt Wochentag-Nummer in deutschen Namen um."""
    namen = ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
             "Freitag", "Samstag", "Sonntag"]
    return namen[nr]

def risiko_farbe(risiko_label):
    """Gibt die passende Farbe für ein Risikolevel zurück."""
    if "Niedrig" in risiko_label: return "success"
    if "Mittel"  in risiko_label: return "warning"
    return "error"
