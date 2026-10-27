"""Schritt 2: Geometrie mit pure functions (reinen Funktionen)."""

import math
from typing import Optional

from pipeline.types import Keypoint, Person

# --- vorgegeben ---------------------------------------------------------------
# Index von Hüfte, Knie, Fussgelenk pro Seite
KNEE_JOINTS = {"left": (11, 13, 15), "right": (12, 14, 16)}


# --- Runde 1: eure Aufgabe ----------------------------------------------------
def to_pixels(p: Person, width: int, height: int) -> Person:
    """Pure. Neue Person mit Pixel-Koordinaten. Die Eingabe p bleibt unverändert.
    Beispiel: Bild 416 x 640, Nase (0.5, 0.25) -> (208.0, 160.0)"""
    cx, cy, w, h = p.box
    box = (cx * width, cy * height, w * width, h * height)
    # TODO: alle 17 Keypoints umrechnen: x * width, y * height, v bleibt gleich.
    #       kps soll ein Tupel von 17 Keypoint sein.
    kps = ...
    raise NotImplementedError("Schritt 2: kps ergänzen, dann diese Zeile löschen")

    return Person(box=box, keypoints=kps, score=p.score)


def knee_angle(p_px: Person, side: str) -> Optional[float]:
    """Pure. Kniewinkel in Grad, oder None.
    side ist "left" (Punkte 11, 13, 15) oder "right" (12, 14, 16)."""
    hip, knee, ankle = (p_px.keypoints[i] for i in KNEE_JOINTS[side])
    # TODO: None zurückgeben, wenn einer der drei Punkte nicht markiert ist (v <= 0)
    # TODO: sonst angle_at(hip, knee, ankle) zurückgeben
    raise NotImplementedError("Schritt 2: knee_angle")


# --- vorgegeben ---------------------------------------------------------------
def angle_at(a: Keypoint, b: Keypoint, c: Keypoint) -> Optional[float]:
    """Pure. Winkel in Grad bei b zwischen den Strecken b->a und b->c (0 bis 180).
    Ist eine der beiden Strecken die Länge 0, gibt es keinen Winkel: None."""
    ux, uy = a.x - b.x, a.y - b.y
    wx, wy = c.x - b.x, c.y - b.y
    lu, lw = math.hypot(ux, uy), math.hypot(wx, wy)
    if lu == 0 or lw == 0:
        return None
    cos = (ux * wx + uy * wy) / (lu * lw)
    cos = max(-1.0, min(1.0, cos))  # Rundungsfehler abfangen
    return math.degrees(math.acos(cos))


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
