"""Schritt 6: Kniebeugen zählen.

Unrein (Schale): frames, angles (Video lesen, Modell aufrufen; beide lazy).
Rein (Kern): drop_missing, ema, step, count_reps, states.
"""

from typing import Iterator, NamedTuple, Optional

from pipeline.inference import WEIGHTS

# --- vorgegeben ---------------------------------------------------------------
DOWN_BELOW = 100  # unter diesem Winkel: tief genug (anpassbar)
UP_ABOVE = 160  # über diesem Winkel: wieder aufgestanden (anpassbar)


class RepState(NamedTuple):
    phase: str  # "up" oder "down"
    count: int


START = RepState("up", 0)


# --- eure Aufgabe -------------------------------------------------------------
def frames(path: str) -> Iterator:
    """Unrein. Generator mit cv2.VideoCapture, liefert ein Bild nach dem anderen.

    Jedes Bild als RGB-numpy-Array (cv2.cvtColor, BGR -> RGB). Mit try/finally
    dafür sorgen, dass cap.release() immer aufgerufen wird.
    """
    raise NotImplementedError


def angles(frames, weights: str = WEIGHTS, side: str = "left") -> Iterator[Optional[float]]:
    """Unrein, lazy (Generator). Pro Bild einen Kniewinkel.

    Pro Bild: Inferenz (predict_image mit Image.fromarray), erste Person,
    to_pixels, knee_angle. Keine Person ergibt None.
    """
    raise NotImplementedError


def drop_missing(xs) -> list[float]:
    """Rein. None weglassen."""
    raise NotImplementedError


def ema(xs: list[float], alpha: float) -> list[float]:
    """Rein. Exponentiell glätten: list(accumulate(xs, lambda s, x: alpha * x + (1 - alpha) * s)).

    Leere Liste ergibt [].
    """
    raise NotImplementedError


def step(state: RepState, angle: float) -> RepState:
    """Rein. Zustandsautomat, gibt einen neuen RepState zurück.

    - phase "up" und angle < DOWN_BELOW: wechselt zu "down" (count bleibt).
    - phase "down" und angle > UP_ABOVE: wechselt zu "up", count + 1.
    - Sonst bleibt der Zustand gleich.
    """
    raise NotImplementedError


def count_reps(xs) -> int:
    """Rein. Anzahl Kniebeugen: reduce(step, xs, START).count"""
    raise NotImplementedError


def states(xs) -> list[RepState]:
    """Rein, optional. Alle Zwischenzustände (für ein Diagramm):
    list(accumulate(xs, step, initial=START))
    """
    raise NotImplementedError
