# Runde 2 – Schritt 1: Unveränderlichkeit und Aliasing.
# Startet mit `uv run python -m bad.b1_clean_inplace`.
# Beobachtet:
#   - Wie viele Keypoints haben die Personen vorher und nachher? Ist kps[13] noch das linke Knie?
#   - Ist `before` auch verändert, obwohl clean(before) nie aufgerufen wurde? Warum?
#   - Was passiert bei der Zuweisung an ds['train'][0]['persons'] und bei p.keypoints[0].x = 1?

from pathlib import Path

from pipeline.data import load_or_build, person_to_row
from pipeline.data import parse_label_file
from pipeline.types import Keypoint, Person

ROOT = Path("data/coco8-pose")

# 1. Labels in eine gewöhnliche Liste von dicts (mit Listen statt Tupeln)
rows = [
    {"file": p.name, "persons": [person_to_row(q) for q in parse_label_file(p.read_text())]}
    for p in sorted((ROOT / "labels" / "train").glob("*.txt"))
]

# 2. Zweite Variable auf dieselbe Liste: kein Kopieren, nur ein zweiter Name
before = rows


# 3. clean entfernt nicht markierte Keypoints direkt aus der Liste
def clean(rows):
    for row in rows:
        for person in row["persons"]:
            for kp in person["keypoints"]:
                if kp[2] == 0:
                    person["keypoints"].remove(kp)  # verändert die Liste!
    return rows


def counts(rows):
    return [len(p["keypoints"]) for row in rows for p in row["persons"]]


print("Vorher (Anzahl Keypoints pro Person):", counts(rows))
clean(rows)

# 4. Ausgabe
print("Nachher (rows):                      ", counts(rows))
print("before ist identisch mit rows?       ", before is rows)
print("Nachher (before):                    ", counts(before))
p0 = rows[0]["persons"][0]
print("kps[13] der ersten Person:           ", p0["keypoints"][13])
for i, person in enumerate(p for row in rows for p in row["persons"]):
    try:
        person["keypoints"][13]
    except IndexError as e:
        print(f"Person {i}: kps[13] -> IndexError: {e} (nur {len(person['keypoints'])} Keypoints)")

# 5. Dasselbe am Dataset versuchen
ds = load_or_build()
print("\nds['train'][0]['persons'] vorher:", len(ds["train"][0]["persons"]), "Personen")
try:
    ds["train"][0]["persons"] = []  # kein Fehler, aber wirkungslos
    print("Zuweisung ohne Fehlermeldung")
except Exception as e:  # pragma: no cover
    print("Zuweisung wirft:", type(e).__name__, e)
print("ds['train'][0]['persons'] nachher:", len(ds["train"][0]["persons"]), "Personen")

# Typen: Person ist ein NamedTuple, also unveränderlich
p = Person(box=(0, 0, 1, 1), keypoints=tuple(Keypoint(0, 0, 0) for _ in range(17)))
try:
    p.keypoints[0].x = 1
except AttributeError as e:
    print("\np.keypoints[0].x = 1 ->", type(e).__name__ + ":", e)
