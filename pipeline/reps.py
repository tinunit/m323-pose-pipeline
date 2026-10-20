"""Schritt 6: Kniebeugen zählen.

Unrein (Schale): frames, angles (Video lesen, Modell aufrufen; beide lazy).
Rein (Kern): drop_missing, ema, step, count_reps, states.
"""

from functools import reduce
from itertools import accumulate
from typing import Iterator, NamedTuple, Optional

import cv2

from pipeline.geometry import knee_angle, to_pixels
from pipeline.inference import WEIGHTS, confident, get_model, predict_image

# --- vorgegeben ---------------------------------------------------------------
DOWN_BELOW = 100  # unter diesem Winkel: tief genug
UP_ABOVE = 160  # über diesem Winkel: wieder aufgestanden


class RepState(NamedTuple):
    phase: str  # "up" oder "down"
    count: int


START = RepState("up", 0)


def frames(path: str) -> Iterator:
    """Generator: liefert ein RGB-Bild (numpy-Array) nach dem anderen.

    Die Datei wird im finally freigegeben, auch wenn der Aufrufer vorzeitig abbricht."""
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def angles(frames, weights: str = WEIGHTS, side: str = "left") -> Iterator[Optional[float]]:
    """Pro Bild: Inferenz, erste Person, to_pixels, knee_angle. Keine Person -> None.

    "Erste Person" = erste Erkennung in der Ausgabe von YOLO (Ultralytics sortiert
    nach Konfidenz absteigend, also die sicherste Erkennung)."""
    from PIL import Image

    get_model(weights)  # Modell einmal laden
    for frame in frames:
        persons = predict_image(Image.fromarray(frame), weights)
        if not persons:
            yield None
        else:
            h, w = frame.shape[:2]
            # unsichere Punkte (verdecktes Knie) -> knee_angle gibt None statt Unsinn
            yield knee_angle(to_pixels(confident(persons[0]), w, h), side)


def drop_missing(xs) -> list[float]:
    """None weglassen."""
    return [x for x in xs if x is not None]


def states(xs) -> list[RepState]:
    """Alle Zwischenzustände (für ein Diagramm)."""
    return list(accumulate(xs, step, initial=START))


# --- Runde 1: eure Aufgabe ----------------------------------------------------
def ema(xs: list[float], alpha: float) -> list[float]:
    """Rein. Exponentielles Glätten einer Folge von Winkeln.

    Der erste Wert bleibt unverändert; jeder weitere ist
    alpha * neuer Wert + (1 - alpha) * vorheriger geglätteter Wert.
    Leere Liste ergibt []. Tipp: ein Scan (Zwischenergebnisse eines Folds),
    siehe itertools.accumulate.
    """
    raise NotImplementedError


def step(state: RepState, angle: float) -> RepState:
    """Rein. Ein Schritt des Zustandsautomaten: alter Zustand + Winkel -> neuer Zustand.

    Phase "up": ist der Winkel kleiner als DOWN_BELOW, wechselt die Phase zu
    "down" (count bleibt). Phase "down": ist der Winkel grösser als UP_ABOVE,
    wechselt die Phase zu "up" und count wird um 1 erhöht. Sonst bleibt der
    Zustand gleich. Der alte Zustand wird nicht verändert.
    """
    raise NotImplementedError


def count_reps(xs) -> int:
    """Rein. Anzahl Wiederholungen in einer Folge von Winkeln.

    Fold von `step` über die Winkel, Start bei START; Ergebnis ist der count
    des Endzustands. Leere Folge ergibt 0.
    """
    raise NotImplementedError
