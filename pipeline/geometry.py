"""Schritt 2: Geometrie mit reinen Funktionen."""

from typing import Optional

from pipeline.types import Keypoint, Person

# vorgegeben: Index von Hüfte, Knie, Fussgelenk pro Seite
KNEE_JOINTS = {"left": (11, 13, 15), "right": (12, 14, 16)}


def to_pixels(p: Person, width: int, height: int) -> Person:
    """Rein. Neue Person mit x * width und y * height. Die Box ebenso. v bleibt gleich.

    Die Eingabe darf nicht verändert werden. Box ist (cx, cy, w, h):
    cx und w mit width, cy und h mit height skalieren.
    """
    raise NotImplementedError


def angle_at(a: Keypoint, b: Keypoint, c: Keypoint) -> Optional[float]:
    """Rein. Winkel in Grad bei b zwischen den Strecken b->a und b->c.

    Formel: cos = (u . w) / (|u| * |w|) mit u = a - b und w = c - b,
    dann math.degrees(math.acos(cos)).
    Den Cosinus vor acos auf den Bereich -1 bis 1 begrenzen (Rundungsfehler).
    None, wenn u oder w die Länge 0 hat.
    """
    raise NotImplementedError


def knee_angle(p_px: Person, side: str) -> Optional[float]:
    """Rein. Kniewinkel einer Seite, berechnet mit Pixelkoordinaten.

    side ist "left" (Hüfte 11, Knie 13, Fussgelenk 15) oder "right" (12, 14, 16),
    siehe KNEE_JOINTS. None, wenn Hüfte, Knie oder Fussgelenk nicht markiert ist (v <= 0).
    """
    raise NotImplementedError
