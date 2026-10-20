"""Schritt 2: Geometrie mit reinen Funktionen."""

import math
from typing import Optional

from pipeline.types import Keypoint, Person

# --- vorgegeben ---------------------------------------------------------------
# Index von Hüfte, Knie, Fussgelenk pro Seite
KNEE_JOINTS = {"left": (11, 13, 15), "right": (12, 14, 16)}


# --- Runde 1: eure Aufgabe ----------------------------------------------------
def to_pixels(p: Person, width: int, height: int) -> Person:
    """Rein. Neue Person in Pixelkoordinaten, die Eingabe bleibt unverändert.

    x-Werte (auch die Box-Mitte cx und die Box-Breite w) werden mit width
    multipliziert, y-Werte (cy, h) mit height. v und score bleiben gleich.
    """
    raise NotImplementedError


def angle_at(a: Keypoint, b: Keypoint, c: Keypoint) -> Optional[float]:
    """Rein. Winkel in Grad bei b zwischen den Strecken b->a und b->c.

    Ergebnis zwischen 0 und 180. Ist eine der beiden Strecken die Länge 0
    (Punkte fallen zusammen), gibt es keinen Winkel: None. Achtung: wegen
    Rundungsfehlern kann der Wert, den ihr an acos übergebt, minimal ausserhalb
    von [-1, 1] liegen.
    """
    raise NotImplementedError


def knee_angle(p_px: Person, side: str) -> Optional[float]:
    """Rein. Kniewinkel einer Seite ("left" oder "right") in Grad.

    Die Person muss in Pixeln vorliegen. Die drei Punkte Hüfte, Knie, Fussgelenk
    stehen in KNEE_JOINTS. Ist einer davon nicht markiert (v <= 0), oder gibt
    angle_at None zurück: None.
    """
    raise NotImplementedError


# --- vorgegeben ---------------------------------------------------------------
if __name__ == "__main__":
    from pipeline.data import load_or_build, row_to_person
    from pipeline.draw import draw_skeleton

    train = load_or_build()["train"]
    saved = False
    for row in train:
        w, h = row["width"], row["height"]
        print(row["file_name"], f"({w}x{h})")
        for i, d in enumerate(row["persons"]):
            p_px = to_pixels(row_to_person(d), w, h)
            left, right = knee_angle(p_px, "left"), knee_angle(p_px, "right")
            print(f"  person {i}: left_knee={left}  right_knee={right}")
        if not saved and row["persons"]:
            img = row["image"].convert("RGB")
            for d in row["persons"]:
                img = draw_skeleton(img, to_pixels(row_to_person(d), w, h))
            img.save("out/skeleton.png")
            saved = True
