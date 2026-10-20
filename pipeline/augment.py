"""Schritt 7 (optional): Bilder spiegeln. Zufall kommt als Argument (Seed) herein."""

import random
from typing import Callable

from PIL import Image

from pipeline.data import person_to_row, row_to_person
from pipeline.types import Keypoint, Person

# --- vorgegeben ---------------------------------------------------------------
# Nach dem Spiegeln tauschen links und rechts die Plätze (Auge 1 <-> Auge 2, ...).
FLIP_INDEX = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15)


# --- Runde 1: eure Aufgabe ----------------------------------------------------
def hflip_keypoint(k: Keypoint) -> Keypoint:
    """Rein. Einen normalisierten Keypoint horizontal spiegeln: x -> 1 - x.

    Nur markierte Punkte (v > 0) werden gespiegelt; nicht markierte Punkte
    (v = 0) bleiben unverändert, sonst entstünde aus (0, 0) ein Punkt bei x = 1.
    """
    raise NotImplementedError


def hflip_person(p: Person) -> Person:
    """Rein. Eine Person horizontal spiegeln (neue Person, Eingabe unverändert).

    Alle Keypoints spiegeln, dann links und rechts neu ordnen (siehe
    FLIP_INDEX: der neue Punkt i ist der gespiegelte Punkt FLIP_INDEX[i]).
    Die Box-Mitte wird gespiegelt (cx -> 1 - cx), cy, w, h und score bleiben.
    """
    raise NotImplementedError


# --- vorgegeben ---------------------------------------------------------------
def hflip_row(row: dict) -> dict:
    """Bild und alle Label-Personen zusammen spiegeln. Die Spalte pred entfällt."""
    return {
        **{k: v for k, v in row.items() if k != "pred"},
        "image": row["image"].transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "persons": [
            person_to_row(hflip_person(row_to_person(d))) for d in row["persons"]
        ],
    }


def seed_for(base_seed: int, epoch: int, idx: int) -> int:
    return base_seed * 1_000_003 + epoch * 1_009 + idx


def random_hflip(row: dict, rng: random.Random) -> dict:
    """Mit Wahrscheinlichkeit 0.5 spiegeln."""
    return hflip_row(row) if rng.random() < 0.5 else row


def make_transform(base_seed: int, epoch: int) -> Callable[[dict], dict]:
    """Funktion für Dataset.with_transform: Batch (dict von Listen) rein, Batch raus."""

    def transform(batch: dict) -> dict:
        # pred überall entfernen, sonst haben gespiegelte und ungespiegelte Zeilen andere Spalten
        cols = {k: v for k, v in batch.items() if k != "pred"}
        rows = [dict(zip(cols, values)) for values in zip(*cols.values())]
        out = [
            random_hflip(row, random.Random(seed_for(base_seed, epoch, row["idx"])))
            for row in rows
        ]
        return {k: [r[k] for r in out] for k in cols}

    return transform
