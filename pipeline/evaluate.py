"""Schritt 5: Auswerten mit fold (reduce) und einem Monoid (Counts, ZERO, add_counts)."""

import math
from typing import NamedTuple, Optional

from pipeline.types import Keypoint, Person

# --- vorgegeben ---------------------------------------------------------------
ALPHA = 0.1  # richtig, wenn Abstand <= ALPHA * Personengrösse


class Counts(NamedTuple):
    correct: int
    total: int


ZERO = Counts(0, 0)  # neutrales Element


def _dist(a: Keypoint, b: Keypoint) -> float:  # vorgegeben
    """Euklidischer Abstand zweier Keypoints."""
    return math.hypot(a.x - b.x, a.y - b.y)


# --- eure Aufgabe -------------------------------------------------------------
def match(gt_px: Person, preds_px: tuple[Person, ...]) -> Optional[Person]:
    """Rein. Die Vorhersage mit dem kleinsten mittleren Abstand zur Label-Person.

    Der Abstand wird über die markierten Label-Punkte (v > 0) gemittelt.
    None, wenn es keine Vorhersage gibt (auch None, wenn die Label-Person keinen
    markierten Punkt hat). Vereinfachung: Eine Vorhersage darf zu zwei
    Label-Personen passen.
    """
    raise NotImplementedError


def person_counts(gt_px: Person, pred_px: Optional[Person], alpha: float) -> Counts:
    """Rein. Richtige und gezählte Punkte einer Label-Person.

    Gezählt werden nur markierte Label-Keypoints. Richtig, wenn der Abstand zum
    vorhergesagten Keypoint höchstens alpha * Personengrösse ist (Personengrösse =
    grössere Seite der Label-Box in Pixeln). Die Konfidenz der Vorhersage spielt
    keine Rolle. Bei pred_px = None: Counts(0, Anzahl markierter Punkte).
    """
    raise NotImplementedError


def add_counts(a: Counts, b: Counts) -> Counts:
    """Rein. Komponentenweise addieren (assoziativ, ZERO ist neutral)."""
    raise NotImplementedError


def image_counts(row: dict, alpha: float) -> Counts:
    """Rein. Counts eines Bildes: reduce(add_counts, ..., ZERO) über alle Label-Personen.

    row hat die Spalten persons (Label), pred (Vorhersage), width, height.
    Beide Seiten mit row_to_person und to_pixels in Pixel umrechnen,
    dann pro Label-Person match und person_counts.
    """
    raise NotImplementedError


def pck(c: Counts) -> Optional[float]:
    """Rein. Anteil richtiger Punkte: correct / total, oder None, wenn total == 0."""
    raise NotImplementedError
