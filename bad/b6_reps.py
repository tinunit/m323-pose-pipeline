# Runde 2 – Schritt 4: State as a value (Zustand als Wert) (kein globaler Zustand).
# Startet mit `uv run python -m bad.b6_reps`.
# Beobachtet:
#   - Was geben der 1. und der 2. Aufruf von count_reps_global mit derselben
#     Eingabe aus? Was erwartet ihr, und was gibt die reine Version aus?
#   - Wo steckt der Zustand, der zwischen den Aufrufen überlebt?
#   - Wie müsste man die Funktion aufrufen, damit sie immer dasselbe liefert?

from pipeline.reps import count_reps

count = 0
phase = "up"


def count_reps_global(xs):
    """FALSCH: Zähler und Phase stehen ausserhalb der Funktion."""
    global count, phase
    for a in xs:
        if phase == "up" and a < 100:
            phase = "down"
        elif phase == "down" and a > 160:
            phase = "up"
            count += 1
    return count


if __name__ == "__main__":
    reps = [170, 90, 165, 85, 170]
    print("global, 1. Aufruf:", count_reps_global(reps))
    print("global, 2. Aufruf:", count_reps_global(reps))
    print("rein  , 1. und 2. Aufruf:", count_reps(reps), count_reps(reps))
