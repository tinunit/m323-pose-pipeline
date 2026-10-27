"""Schritt 4: Kniebeugen zählen.

Impure (Schale): frames, angles (Video lesen, Modell aufrufen; beide lazy).
Pure (Kern): drop_missing, ema, step, count_reps, states.
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


def ema(xs: list[float], alpha: float) -> list[float]:
    """Pure. Exponentielles Glätten (Scan). Leere Liste -> []."""
    return list(accumulate(xs, lambda s, x: alpha * x + (1 - alpha) * s))


def states(xs) -> list[RepState]:
    """Alle Zwischenzustände (für ein Diagramm)."""
    return list(accumulate(xs, step, initial=START))


# --- Runde 1: eure Aufgabe ----------------------------------------------------
def step(state: RepState, angle: float) -> RepState:
    """Pure. Ein Schritt des Zustandsautomaten: alter Zustand + Winkel -> neuer Zustand."""
    # Gegeben: von "up" nach "down", sobald das Knie tief genug gebeugt ist (Winkel < DOWN_BELOW)
    if state.phase == "up" and angle < DOWN_BELOW:
        return RepState("down", state.count)
    # TODO: von "down" zurück nach "up", sobald der Winkel > UP_ABOVE ist.
    #       Damit ist eine Wiederholung fertig: neuer Zustand mit count + 1.
    #       Beispiel: step(RepState("down", 2), 170) == RepState("up", 3)
    # TODO: in allen anderen Fällen den Zustand unverändert zurückgeben.
    #       Beispiel: step(RepState("up", 2), 130) == RepState("up", 2)
    raise NotImplementedError("Schritt 4: step")


def count_reps(xs: list[float]) -> int:
    """Pure. Anzahl Wiederholungen aus einer Folge von Kniewinkeln (ein Winkel pro Videobild)."""
    # TODO: step der Reihe nach auf alle Winkel in xs anwenden, beginnend bei START,
    #       und am Schluss die Anzahl Wiederholungen (.count) zurückgeben.
    #       Tipp: functools.reduce(funktion, liste, startwert)
    #       Beispiel: count_reps([170, 90, 165, 85, 170]) == 2
    raise NotImplementedError("Schritt 4: count_reps")
