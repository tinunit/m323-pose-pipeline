"""Schritt 3: Filtern mit Funktionen höherer Ordnung."""

from typing import Callable, Optional

from pipeline.data import row_to_person
from pipeline.geometry import knee_angle
from pipeline.types import Person


# --- vorgegeben ---------------------------------------------------------------
def labelled(p: Person) -> int:
    """Anzahl markierter Keypoints (v > 0)."""
    return sum(1 for k in p.keypoints if k.v > 0)


# --- Runde 1: eure Aufgabe ----------------------------------------------------
def min_labelled(n: int) -> Callable[[Person], bool]:
    """Gibt ein Prädikat zurück: hat die Person mindestens n markierte Keypoints?

    Funktion höherer Ordnung: min_labelled(8) ist selbst eine Funktion
    Person -> bool. Markiert heisst v > 0 (siehe `labelled`).
    """
    raise NotImplementedError


def any_person(pred: Callable[[Person], bool]) -> Callable[[dict], bool]:
    """Wandelt ein Personen-Prädikat in ein Bild-Prädikat um.

    Das Ergebnis nimmt eine Dataset-Zeile (dict mit Schlüssel "persons", eine
    Liste von Zeilen-dicts; mit row_to_person in Personen umwandeln) und ist
    wahr, wenn mindestens eine Person pred erfüllt. Ohne Personen: falsch.
    """
    raise NotImplementedError


def all_of(*preds: Callable) -> Callable:
    """Kombiniert beliebig viele Prädikate zu einem: wahr, wenn alle wahr sind.

    Das Ergebnis nimmt dasselbe Argument wie die einzelnen Prädikate. Ohne
    Argumente ist es immer wahr.
    """
    raise NotImplementedError


# --- vorgegeben ---------------------------------------------------------------
def max_persons(k: int) -> Callable[[dict], bool]:
    """Bild-Prädikat: höchstens k Personen."""
    return lambda row: len(row["persons"]) <= k


def knee_angles(persons_px: tuple[Person, ...]) -> tuple[float, ...]:
    """Alle berechenbaren Kniewinkel (links und rechts); None fällt weg (mapMaybe)."""
    angles = (
        knee_angle(p, side) for p in persons_px for side in ("left", "right")
    )
    return tuple(a for a in angles if a is not None)


def angle_or(value: Optional[float], default: float) -> float:
    """value, oder default bei None (fromMaybe)."""
    return default if value is None else value


if __name__ == "__main__":
    from pipeline.data import load_or_build
    from pipeline.geometry import to_pixels

    ds = load_or_build()
    train = ds["train"]
    selected = train.filter(all_of(any_person(min_labelled(8)), max_persons(3)))
    print(f"Bilder vorher: {len(train)}, nachher: {len(selected)}")

    persons_px = tuple(
        to_pixels(row_to_person(d), row["width"], row["height"])
        for row in selected
        for d in row["persons"]
    )
    angles = knee_angles(persons_px)
    print(f"Kniewinkel ({len(angles)}):")
    print([round(a, 1) for a in angles])
