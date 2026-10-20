"""Schritt 3: Filtern mit Funktionen höherer Ordnung."""

from typing import Callable, Optional

from pipeline.types import Person


def labelled(p: Person) -> int:
    """Rein. Anzahl markierter Keypoints (v > 0)."""
    raise NotImplementedError


def min_labelled(n: int) -> Callable[[Person], bool]:
    """Rein. Gibt ein Prädikat zurück: hat die Person mindestens n markierte Keypoints?

    Achtung Closure: Jeder Aufruf soll sein eigenes n behalten (siehe bad/b3_late_binding.py).
    """
    raise NotImplementedError


def any_person(pred: Callable[[Person], bool]) -> Callable[[dict], bool]:
    """Rein. Bild-Prädikat: mindestens eine Person des Bildes erfüllt pred.

    Das Prädikat bekommt eine Dataset-Zeile (dict); die Personen stehen in
    row["persons"] als dicts (row_to_person aus pipeline.data).
    """
    raise NotImplementedError


def max_persons(k: int) -> Callable[[dict], bool]:
    """Rein. Bild-Prädikat: höchstens k Personen im Bild."""
    raise NotImplementedError


def all_of(*preds: Callable) -> Callable:
    """Rein. Wahr, wenn alle Prädikate wahr sind. all_of() ohne Argumente ist immer wahr."""
    raise NotImplementedError


def knee_angles(persons_px: tuple[Person, ...]) -> tuple[float, ...]:
    """Rein. Alle berechenbaren Kniewinkel (links und rechts) der Personen in Pixeln.

    None wird weggelassen (mapMaybe).
    """
    raise NotImplementedError


def angle_or(value: Optional[float], default: float) -> float:
    """Rein. value, oder default bei None (fromMaybe)."""
    raise NotImplementedError
