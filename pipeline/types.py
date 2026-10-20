from typing import NamedTuple


class Keypoint(NamedTuple):
    x: float
    y: float
    v: float  # Label: 0, 1 oder 2. Vorhersage: Konfidenz 0..1


class Person(NamedTuple):
    box: tuple[float, float, float, float]  # cx, cy, w, h
    keypoints: tuple[Keypoint, ...]  # immer genau 17
    score: float = 1.0  # Label: 1.0. Vorhersage: Konfidenz
