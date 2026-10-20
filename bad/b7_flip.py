# Runde 2 – Schritt 7 (Bonus): Tests und Properties, die zu schwach sind.
# Startet mit `uv run python -m bad.b7_flip`.
# Beobachtet:
#   (a) Spiegeln ohne links/rechts zu tauschen
#       - Was ist in out/b7_falsch.png anders als in out/b7_richtig.png
#         (achtet auf die Farben der beiden Körperseiten)?
#       - Wie verändert sich der PCK gegen die falschen Labels, bei ALPHA 0.1
#         und bei 0.05?
#   (b) Der Test hflip(hflip(p)) == p
#       - Besteht er auch für die falsche Version? Was sagt das über den Test?
#       - Welche zusätzliche Eigenschaft müsste ein Test prüfen, damit die
#         falsche Version durchfällt?

from pathlib import Path

from PIL import Image

from pipeline.augment import hflip_keypoint, hflip_person
from pipeline.data import load_or_build, person_to_row, row_to_person
from pipeline.draw import draw_skeleton
from pipeline.geometry import to_pixels
from pipeline.types import Person

OUT = Path("out")
FLIP = Image.Transpose.FLIP_LEFT_RIGHT


def hflip_person_wrong(p: Person) -> Person:
    """FALSCH: x wird gespiegelt, aber links und rechts bleiben an ihrem Index."""
    cx, cy, w, h = p.box
    return Person((1 - cx, cy, w, h), tuple(hflip_keypoint(k) for k in p.keypoints), p.score)


# ---------------------------------------------------------------- (a)
def part_a() -> None:
    print("=== (a) Spiegeln ohne links/rechts zu tauschen ===")
    OUT.mkdir(exist_ok=True)
    ds = load_or_build()["val"]
    row = ds[0]
    flipped = row["image"].transpose(FLIP)
    gt = row_to_person(row["persons"][0])
    w, h = flipped.size
    for name, fn in (("richtig", hflip_person), ("falsch", hflip_person_wrong)):
        img = draw_skeleton(flipped, to_pixels(fn(gt), w, h))
        img.save(OUT / f"b7_{name}.png")
    print("Skelette gespeichert: out/b7_richtig.png, out/b7_falsch.png\n")
    pck_comparison(ds, alphas=(0.1, 0.05))


def pck_comparison(ds, alphas) -> None:
    from pipeline import evaluate
    from pipeline.inference import predict_image

    results = {}
    for name, fn in (("richtig", hflip_person), ("falsch", hflip_person_wrong)):
        for alpha in alphas:
            total = evaluate.ZERO
            for row in ds:
                preds = predict_image(row["image"].transpose(FLIP))
                r = {
                    "width": row["width"],
                    "height": row["height"],
                    "persons": [
                        person_to_row(fn(row_to_person(d))) for d in row["persons"]
                    ],
                    "pred": [person_to_row(p) for p in preds],
                }
                total = evaluate.add_counts(total, evaluate.image_counts(r, alpha))
            results[(name, alpha)] = evaluate.pck(total)
    print("PCK auf gespiegelten Val-Bildern (Inferenz gegen gespiegelte Labels):")
    for (name, alpha), v in results.items():
        print(f"  Labels {name:8s} ALPHA={alpha}: PCK = {v:.3f}")
    print()


# ---------------------------------------------------------------- (b)
def part_b() -> None:
    print("=== (b) Der Test hflip(hflip(p)) == p ===")
    ds = load_or_build()["val"]
    ps = [row_to_person(d) for row in ds for d in row["persons"]]
    ok_right = all(hflip_person(hflip_person(p)) == p for p in ps)
    ok_wrong = all(
        all(
            abs(a.x - b.x) < 1e-6 and a.y == b.y and a.v == b.v
            for a, b in zip(hflip_person_wrong(hflip_person_wrong(p)).keypoints, p.keypoints)
        )
        for p in ps
    )
    print(f"richtige Version besteht den Test: {ok_right}")
    print(f"FALSCHE Version besteht den Test:  {ok_wrong}\n")


if __name__ == "__main__":
    part_a()
    part_b()
