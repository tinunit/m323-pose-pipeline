"""Schritt 1: Daten laden."""

from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, Features, Image, Value

from pipeline.types import Keypoint, Person

# --- vorgegeben ---------------------------------------------------------------
N_KEYPOINTS = 17
N_VALUES = 5 + 3 * N_KEYPOINTS  # 56 Zahlen pro Zeile

PERSON_FEATURE = {
    "box": [Value("float32")],
    "keypoints": [[Value("float32")]],
    "score": Value("float32"),
}

FEATURES = Features({
    "file_name": Value("string"),
    "image": Image(),
    "width": Value("int32"),
    "height": Value("int32"),
    "persons": [PERSON_FEATURE],
})

DATA_ROOT = Path("data/coco8-pose")
DISK_PATH = Path("data/coco8-pose-ds")


def labelled_count(p: Person) -> int:  # vorgegeben
    """Anzahl markierter Keypoints (v > 0)."""
    return sum(1 for k in p.keypoints if k.v > 0)


# --- eure Aufgabe -------------------------------------------------------------
def parse_label_line(line: str) -> Optional[Person]:
    """Rein. Eine Label-Zeile in eine Person umwandeln.

    Format: Klasse, cx, cy, w, h, dann 17 mal (x, y, v). Genau 56 Zahlen.
    Gibt None zurück, wenn die Zeile nicht genau 56 Zahlen hat.
    Tipp: Tupel statt Listen verwenden (Unveränderlichkeit).
    """
    raise NotImplementedError


def parse_label_file(text: str) -> tuple[Person, ...]:
    """Rein. Ganze Label-Datei parsen.

    Eine leere Datei ergibt (). Ungültige Zeilen werden weggelassen.
    """
    raise NotImplementedError


def person_to_row(p: Person) -> dict:
    """Rein. Person in das Dataset-Format umwandeln (dict mit Listen statt Tupeln).

    Schlüssel: "box", "keypoints", "score" (siehe PERSON_FEATURE).
    """
    raise NotImplementedError


def row_to_person(d: dict) -> Person:
    """Rein. Gegenstück zu person_to_row: dict zurück in eine unveränderliche Person.

    Es muss gelten: row_to_person(person_to_row(p)) == p
    """
    raise NotImplementedError


def load_split(root: Path, split: str) -> Dataset:
    """Unrein (liest von der Festplatte). Bilder und Labels eines Splits laden.

    Liest root/images/{split}/*.jpg und root/labels/{split}/*.txt (sortiert).
    Pro Bild eine Zeile mit den Spalten aus FEATURES; "persons" enthält
    person_to_row(...) für jede gültige Label-Zeile. Fehlt die Label-Datei,
    gibt es keine Personen. Rückgabe: Dataset mit features=FEATURES.
    """
    raise NotImplementedError


# --- vorgegeben (Plumbing) -----------------------------------------------------
def load_all(root: Path) -> DatasetDict:  # vorgegeben
    return DatasetDict({s: load_split(root, s) for s in ("train", "val")})


def load_or_build(root: Path = DATA_ROOT, disk: Path = DISK_PATH) -> DatasetDict:  # vorgegeben
    """Einmal bauen, mit save_to_disk speichern, danach mit load_from_disk laden.

    Nur ein Dataset von der Festplatte hat Cache-Dateien, die map wiederverwenden kann.
    """
    from datasets import load_from_disk

    if not disk.exists():
        load_all(root).save_to_disk(str(disk))
    return load_from_disk(str(disk))


def summary(ds: Dataset) -> list[tuple[str, int, int]]:  # vorgegeben
    """Pro Bild: Dateiname, Anzahl Personen, Anzahl markierte Keypoints."""
    return [
        (
            row["file_name"],
            len(row["persons"]),
            sum(labelled_count(row_to_person(p)) for p in row["persons"]),
        )
        for row in ds
    ]


if __name__ == "__main__":
    dsd = load_or_build()
    print(dsd)
    for split in ("train", "val"):
        print(f"\n{split}")
        for name, n, k in summary(dsd[split]):
            print(f"  {name}  persons={n}  keypoints={k}")
