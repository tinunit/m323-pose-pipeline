# Runde 2 – Schritt 4: Reiner Kern, unreine Schale und Memoisierung.
# Startet mit `uv run python -m bad.b4_stale_cache`.
# Beobachtet:
#   - Was geben die beiden ersten Aufrufe von count_people aus, obwohl CONF dazwischen geändert wurde?
#   - Woraus besteht der Schlüssel von lru_cache? Was gehört nicht dazu?
#   - Warum ist die Variante mit conf als Argument korrekt?

from functools import lru_cache

from ultralytics import YOLO

CONF = 0.25
MODEL = YOLO("yolo11n-pose.pt")


@lru_cache
def count_people(path: str) -> int:
    results = MODEL.predict(path, conf=CONF, verbose=False)
    return len(results[0].boxes)


IMG = "data/coco8-pose/images/val/000000000110.jpg"
print(count_people(IMG))
CONF = 0.9
print(count_people(IMG))


# --- Korrigierte Version: conf ist ein Argument und damit Teil des Cache-Schlüssels ---
@lru_cache
def count_people_fixed(path: str, conf: float) -> int:
    results = MODEL.predict(path, conf=conf, verbose=False)
    return len(results[0].boxes)


print("fixed:", count_people_fixed(IMG, 0.25), count_people_fixed(IMG, 0.9))
