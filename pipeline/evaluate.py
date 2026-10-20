"""Schritt 5: Auswerten mit fold (reduce) und einem Monoid (Counts, ZERO, add_counts)."""

import math
from functools import partial, reduce
from typing import NamedTuple, Optional

from pipeline.data import row_to_person
from pipeline.filters import labelled
from pipeline.geometry import to_pixels
from pipeline.types import Keypoint, Person

# --- vorgegeben ---------------------------------------------------------------
ALPHA = 0.1  # richtig, wenn Abstand <= ALPHA * Personengrösse


class Counts(NamedTuple):
    correct: int
    total: int


ZERO = Counts(0, 0)  # neutrales Element


def _dist(a: Keypoint, b: Keypoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _mean_distance(gt_px: Person, pred_px: Person) -> float:
    """Mittlerer Abstand über die markierten Label-Punkte (v > 0)."""
    ds = [_dist(g, p) for g, p in zip(gt_px.keypoints, pred_px.keypoints) if g.v > 0]
    return sum(ds) / len(ds)


def match(gt_px: Person, preds_px: tuple[Person, ...]) -> Optional[Person]:
    """Vorhersage mit dem kleinsten mittleren Abstand zu den markierten Label-Punkten.

    None, wenn es keine Vorhersage gibt. Hat die Label-Person keinen markierten
    Punkt, gibt es nichts zu vergleichen: ebenfalls None (-> Counts(0, 0)).
    """
    if not preds_px or labelled(gt_px) == 0:
        return None
    return min(preds_px, key=partial(_mean_distance, gt_px))


def image_counts(row: dict, alpha: float) -> Counts:
    """Counts eines Bildes: fold über alle Label-Personen."""
    w, h = row["width"], row["height"]
    gts = tuple(to_pixels(row_to_person(d), w, h) for d in row["persons"])
    preds = tuple(to_pixels(row_to_person(d), w, h) for d in row["pred"])
    return reduce(
        add_counts,
        (person_counts(gt, match(gt, preds), alpha) for gt in gts),
        ZERO,
    )


# --- Runde 1: eure Aufgabe ----------------------------------------------------
def person_counts(gt_px: Person, pred_px: Optional[Person], alpha: float) -> Counts:
    """Rein. Richtige und gezählte Punkte einer Label-Person.

    Gezählt werden nur markierte Label-Keypoints (v > 0). Richtig ist ein
    Punkt, wenn sein Abstand (`_dist`) zum gleich nummerierten vorhergesagten
    Keypoint höchstens alpha * Personengrösse ist. Personengrösse = grössere
    Seite der Label-Box in Pixeln (Breite oder Höhe). Die Konfidenz der
    Vorhersage spielt keine Rolle. Bei pred_px = None: Counts(0, Anzahl
    markierter Punkte).
    """
    raise NotImplementedError


def add_counts(a: Counts, b: Counts) -> Counts:
    """Rein. Zwei Counts komponentenweise addieren.

    Die Operation ist assoziativ und ZERO ist das neutrale Element, damit
    sie sich für reduce eignet.
    """
    raise NotImplementedError


def pck(c: Counts) -> Optional[float]:
    """Rein. Anteil richtiger Punkte (correct / total).

    None, wenn total == 0 (nichts gezählt).
    """
    raise NotImplementedError
