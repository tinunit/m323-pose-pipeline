from typing import NamedTuple


class Keypoint(NamedTuple):
    """Ein Gelenkpunkt, z.B. die Nase: Keypoint(x=0.520, y=0.381, v=2.0)."""

    x: float  # waagrecht, Anteil der Bildbreite (0 = linker Rand, 1 = rechter Rand). 0.520 = etwas rechts der Mitte
    y: float  # senkrecht, Anteil der Bildhöhe (0 = oberer Rand, 1 = unterer Rand). 0.381 = im oberen Drittel
    v: float  # Label: 0 = nicht markiert, 1 = markiert aber verdeckt, 2 = sichtbar. Vorhersage: Konfidenz 0..1, z.B. 0.93


class Person(NamedTuple):
    box: tuple[float, float, float, float]  # cx, cy, w, h
    keypoints: tuple[Keypoint, ...]  # immer genau 17
    score: float = 1.0  # Label: 1.0. Vorhersage: Konfidenz
