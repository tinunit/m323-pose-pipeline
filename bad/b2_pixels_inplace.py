# Runde 2 – Schritt 2: Reine Funktionen.
# Startet mit `uv run python -m bad.b2_pixels_inplace`.
# Beobachtet:
#   - Wie ändern sich die Koordinaten und der Winkel beim 1. und beim 2. Aufruf?
#   - Was passiert, wenn ihr die Reihenfolge (Winkel, Zeichnen) vertauscht?
#   - Warum verschwindet der Fehler, wenn ihr ds['train'][0] vor jedem Aufruf neu holt?

from pipeline.data import load_or_build, row_to_person
from pipeline.geometry import knee_angle, to_pixels


def to_pixels_inplace(person_dict: dict, w: int, h: int) -> None:
    """FALSCH: überschreibt die Koordinaten in der Liste."""
    person_dict["box"][0] *= w
    person_dict["box"][1] *= h
    person_dict["box"][2] *= w
    person_dict["box"][3] *= h
    for kp in person_dict["keypoints"]:
        kp[0] *= w
        kp[1] *= h


def draw_skeleton_dict(ex: dict):
    for d in ex["persons"]:
        to_pixels_inplace(d, ex["width"], ex["height"])
    # (Zeichnen wie in draw.py; hier nicht nötig, die Mutation ist das Thema)


def knee_angle_dict(ex: dict):
    d = ex["persons"][0]
    to_pixels_inplace(d, ex["width"], ex["height"])
    return knee_angle(row_to_person(d), "left")


def _kp(ex: dict):
    """Linkes Knie (Index 13) der ersten Person, wie es gerade in ex steht."""
    return ex["persons"][0]["keypoints"][13][:2]


def _has_left_leg(d: dict) -> bool:
    return all(d["keypoints"][j][2] > 0 for j in (11, 13, 15))


if __name__ == "__main__":
    train = load_or_build()["train"]
    idx = next(i for i, r in enumerate(train) if r["persons"] and _has_left_leg(r["persons"][0]))
    w, h = train[idx]["width"], train[idx]["height"]
    print(f"Bild {idx}: {train[idx]['file_name']}, {w}x{h}")

    print("\n--- Wie in der Aufgabe: ex einmal holen, zeichnen, dann Winkel ---")
    ex = train[idx]
    print("linkes Knie original (normalisiert):", _kp(ex))
    print("richtiger Winkel (rein, einmal Pixel):", knee_angle(to_pixels(row_to_person(ex["persons"][0]), w, h), "left"))
    draw_skeleton_dict(ex)
    print("nach 1. Aufruf:", _kp(ex))
    angle = knee_angle_dict(ex)
    print("nach 2. Aufruf:", _kp(ex))
    print("Winkel (doppelt skaliert):", angle)

    print("\n--- Zeilen vertauscht: erst Winkel, dann zeichnen ---")
    ex = train[idx]
    print("Winkel:", knee_angle_dict(ex), " Koordinaten:", _kp(ex))
    draw_skeleton_dict(ex)
    print("Koordinaten nach dem Zeichnen (falsch!):", _kp(ex))

    print("\n--- ds['train'][0] vor jedem Aufruf neu holen ---")
    a1, a2 = knee_angle_dict(train[idx]), knee_angle_dict(train[idx])
    print("Winkel 1:", a1, " Winkel 2:", a2, " gleich:", a1 == a2)

    print("\n--- Wie stark ändert sich der Winkel (alle Personen mit linkem Bein)? ---")
    for r_i, r in enumerate(train):
        for p_i, d in enumerate(r["persons"]):
            if not _has_left_leg(d):
                continue
            fresh = lambda: {**r, "persons": [train[r_i]["persons"][p_i]]}
            once = knee_angle_dict(fresh())
            ex = fresh()
            draw_skeleton_dict(ex)
            twice = knee_angle_dict(ex)
            print(f"  {r['file_name']} p{p_i} ({r['width']}x{r['height']}): "
                  f"einmal={once:.1f}  zweimal={twice:.1f}  Differenz={twice - once:+.1f}")
