"""Schritt 1: Daten laden."""

from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

from pipeline.types import Keypoint, Person

# --- vorgegeben ---------------------------------------------------------------
N_KEYPOINTS = 17
N_VALUES = 5 + 3 * N_KEYPOINTS  # 56

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


# --- Runde 1: eure Aufgabe ----------------------------------------------------
def parse_label_line(line: str) -> Optional[Person]:
    """Rein. Eine Zeile im YOLO-Pose-Format in eine Person umwandeln.

    Eine gültige Zeile hat N_VALUES (56) Zahlen, durch Leerzeichen getrennt:
    Klasse, Box (cx cy w h), dann 17 mal (x y v). Die Klasse wird nicht
    gebraucht. Die Box wird als 4er-Tupel übernommen, die Keypoints als Tupel
    von 17 `Keypoint`. Das Feld `score` der Person behält seinen Standardwert.

    Gibt None zurück, wenn die Anzahl Werte nicht stimmt oder sich ein Wert
    nicht in eine Zahl umwandeln lässt.
    """
    raise NotImplementedError


def parse_label_file(text: str) -> tuple[Person, ...]:
    """Rein. Alle gültigen Zeilen einer Label-Datei als Tupel von Personen.

    Ungültige Zeilen (parse_label_line gibt None) und Leerzeilen werden
    weggelassen. Leerer Text ergibt ().
    """
    raise NotImplementedError


# --- vorgegeben ---------------------------------------------------------------


def person_to_row(p: Person) -> dict:
    return {
        "box": list(p.box),
        "keypoints": [list(k) for k in p.keypoints],
        "score": p.score,
    }


def row_to_person(d: dict) -> Person:
    return Person(
        box=tuple(d["box"]),
        keypoints=tuple(Keypoint(*k) for k in d["keypoints"]),
        score=d["score"],
    )


def labelled_count(p: Person) -> int:
    return sum(1 for k in p.keypoints if k.v > 0)


def _example(img_path: Path, label_dir: Path) -> dict:
    """Unrein: liest Bild und Label von der Festplatte."""
    label_path = label_dir / (img_path.stem + ".txt")
    text = label_path.read_text() if label_path.exists() else ""
    with PILImage.open(img_path) as im:
        width, height = im.size
    return {
        "file_name": img_path.name,
        "image": str(img_path),
        "width": width,
        "height": height,
        "persons": [person_to_row(p) for p in parse_label_file(text)],
    }


def load_split(root: Path, split: str) -> Dataset:
    """Liest images/{split}/*.jpg und labels/{split}/*.txt."""
    root = Path(root)
    images = sorted((root / "images" / split).glob("*.jpg"))
    rows = [_example(p, root / "labels" / split) for p in images]
    return Dataset.from_list(rows, features=FEATURES)


def load_all(root: Path) -> DatasetDict:
    return DatasetDict({s: load_split(root, s) for s in ("train", "val")})


DATA_ROOT = Path("data/coco8-pose")
DISK_PATH = Path("data/coco8-pose-ds")


def load_or_build(root: Path = DATA_ROOT, disk: Path = DISK_PATH) -> DatasetDict:
    """Einmal bauen, mit save_to_disk speichern, danach mit load_from_disk laden.

    Nur ein Dataset von der Festplatte hat Cache-Dateien, die map wiederverwenden kann.
    """
    from datasets import load_from_disk

    if not disk.exists():
        load_all(root).save_to_disk(str(disk))
    return load_from_disk(str(disk))


def summary(ds: Dataset) -> list[tuple[str, int, int]]:
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
