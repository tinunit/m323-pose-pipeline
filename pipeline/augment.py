"""Schritt 7 (optional, Bonus): Bilder spiegeln. Zufall kommt als Argument (Seed) herein."""

import random
from typing import Callable

from pipeline.types import Keypoint, Person

# --- vorgegeben ---------------------------------------------------------------
# Nach dem Spiegeln tauschen links und rechts die Plätze (Auge 1 <-> Auge 2, ...).
FLIP_INDEX = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)


# --- eure Aufgabe -------------------------------------------------------------
def hflip_keypoint(k: Keypoint) -> Keypoint:
    """Rein. Markierter Punkt (v > 0): Keypoint(1 - x, y, v). Nicht markiert: unverändert."""
    raise NotImplementedError


def hflip_person(p: Person) -> Person:
    """Rein. Alle Keypoints spiegeln, dann neu ordnen: neu[i] = gespiegelt[FLIP_INDEX[i]].

    Box: cx wird 1 - cx (cy, w, h bleiben). score bleibt.
    """
    raise NotImplementedError


def hflip_row(row: dict) -> dict:
    """Rein. Ein ganzes Dataset-Beispiel spiegeln, Pixel und Labels zusammen.

    Bild mit image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) (PIL), alle
    Label-Personen mit hflip_person (row_to_person / person_to_row).
    Die Spalte "pred" wird entfernt, weil sie neu berechnet werden muss.
    Die Eingabe darf nicht verändert werden.
    """
    raise NotImplementedError


def seed_for(base_seed: int, epoch: int, idx: int) -> int:
    """Rein. base_seed * 1_000_003 + epoch * 1_009 + idx"""
    raise NotImplementedError


def random_hflip(row: dict, rng: random.Random) -> dict:
    """Rein (gegeben rng). Mit Wahrscheinlichkeit 0.5 spiegeln (rng.random() < 0.5)."""
    raise NotImplementedError


def make_transform(base_seed: int, epoch: int) -> Callable[[dict], dict]:
    """Gibt eine Funktion für Dataset.with_transform zurück.

    Die Funktion bekommt einen Batch (dict von Listen), zerlegt ihn in Zeilen,
    wendet random_hflip mit random.Random(seed_for(base_seed, epoch, row["idx"]))
    an und setzt den Batch wieder zu einem dict von Listen zusammen.
    Tipp: Spalte "pred" im Batch weglassen, damit alle Zeilen dieselben Spalten haben.
    Voraussetzung: das Dataset hat eine Spalte "idx"
    (ds.map(lambda row, i: {"idx": i}, with_indices=True)).
    """
    raise NotImplementedError
