"""Auswerten mit fold (reduce) und einem Monoid (Counts, ZERO, add_counts) (vorgegeben)."""

import math
from functools import partial, reduce
from typing import NamedTuple, Optional

from pipeline.data import row_to_person
from pipeline.filters import labelled
from pipeline.geometry import to_pixels
from pipeline.types import Keypoint, Person

ALPHA = 0.1


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


def person_counts(gt_px: Person, pred_px: Optional[Person], alpha: float) -> Counts:
    """Richtige und gezählte Punkte einer Label-Person."""
    marked = [i for i, k in enumerate(gt_px.keypoints) if k.v > 0]
    if pred_px is None:
        return Counts(0, len(marked))
    size = max(gt_px.box[2], gt_px.box[3])  # grössere Seite der Label-Box (Pixel)
    correct = sum(
        1 for i in marked if _dist(gt_px.keypoints[i], pred_px.keypoints[i]) <= alpha * size
    )
    return Counts(correct, len(marked))


def add_counts(a: Counts, b: Counts) -> Counts:
    """Komponentenweise addieren (assoziativ, ZERO ist neutral)."""
    return Counts(a.correct + b.correct, a.total + b.total)


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


def pck(c: Counts) -> Optional[float]:
    """Anteil richtiger Punkte, None bei total == 0."""
    return None if c.total == 0 else c.correct / c.total
