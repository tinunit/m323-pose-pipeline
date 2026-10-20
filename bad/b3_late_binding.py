# Runde 2 – Schritt 3: Funktionen als Werte und Closures.
# Startet mit `uv run python -m bad.b3_late_binding`.
# Beobachtet:
#   - Welche Zahlen gibt die 'falsche' Zeile aus, welche die 'richtige'? Was fällt auf?
#   - Welchen Wert von n benutzen alle Lambdas in der Schleife, und wann wird er nachgeschlagen?
#   - Warum hat min_labelled(n) das Problem nicht?

from pipeline.data import load_or_build, row_to_person
from pipeline.filters import labelled, min_labelled

ds = load_or_build()
# "aller Bilder": train und val zusammen
all_persons = tuple(
    row_to_person(d) for split in ("train", "val") for row in ds[split] for d in row["persons"]
)

THRESHOLDS = (3, 8, 12)

# FALSCH: alle Lambdas teilen sich dieselbe Variable n der Comprehension
preds = [lambda p: labelled(p) >= n for n in THRESHOLDS]
print("falsch :", [sum(1 for p in all_persons if pred(p)) for pred in preds])

# RICHTIG: min_labelled(n) bindet n pro Aufruf neu
preds_ok = [min_labelled(n) for n in THRESHOLDS]
print("richtig:", [sum(1 for p in all_persons if pred(p)) for pred in preds_ok])
