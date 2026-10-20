"""Gesamtlauf: Laden, Filtern, Inferenz, Auswertung.

Start: uv run python run_pipeline.py
"""

from functools import partial, reduce  # noqa: F401  (für die Auswertung)

from pipeline.data import load_or_build  # noqa: F401
from pipeline.evaluate import ALPHA, ZERO, add_counts, image_counts, pck  # noqa: F401
from pipeline.filters import all_of, any_person, max_persons, min_labelled  # noqa: F401
from pipeline.inference import predict_split  # noqa: F401


def fmt(x):  # vorgegeben
    return "  -  " if x is None else f"{x:.3f}"


def main() -> None:
    # TODO 1. Laden: load_or_build() liefert ein DatasetDict mit "train" und "val".
    # TODO 2. Filtern: auf dem train-Split
    #         train.filter(all_of(any_person(min_labelled(8)), max_persons(3)))
    #         und die Anzahl Bilder vor und nach dem Filter ausgeben.
    # TODO 3. Inferenz: predict_split(val) (map mit Zwischenspeicher).
    # TODO 4. Auswertung: score = partial(image_counts, alpha=ALPHA)
    #         - Tabelle pro Bild: file_name, correct, total, pck (fmt(...) für None)
    #         - Gesamt: reduce(add_counts, map(score, ds_pred), ZERO) und Gesamt-PCK
    raise NotImplementedError("run_pipeline.py: TODO-Liste in main() abarbeiten")


if __name__ == "__main__":
    main()
