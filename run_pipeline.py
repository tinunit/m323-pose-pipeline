"""Gesamtlauf: Laden, Filtern, Inferenz, Auswertung.

Start: uv run python run_pipeline.py
"""

from functools import partial, reduce

from pipeline.data import load_or_build
from pipeline.evaluate import ALPHA, ZERO, add_counts, image_counts, pck
from pipeline.filters import all_of, any_person, max_persons, min_labelled
from pipeline.inference import predict_split


def fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


def main() -> None:
    ds = load_or_build()                                   # 1. Laden
    train = ds["train"]
    selected = train.filter(all_of(any_person(min_labelled(8)), max_persons(3)))
    print(f"Filter (train): {len(train)} -> {len(selected)} Bilder")  # 2. Filtern

    val = ds["val"]
    ds_pred = predict_split(val)                           # 3. Inferenz (val)

    score = partial(image_counts, alpha=ALPHA)             # 4. Auswertung
    print(f"\n{'file':<20}{'correct':>8}{'total':>7}{'pck':>8}")
    for row in ds_pred:
        c = score(row)
        print(f"{row['file_name']:<20}{c.correct:>8}{c.total:>7}{fmt(pck(c)):>8}")
    total = reduce(add_counts, map(score, ds_pred), ZERO)
    print(f"\nGesamt: {total.correct}/{total.total}  PCK@{ALPHA} = {fmt(pck(total))}")


if __name__ == "__main__":
    main()
