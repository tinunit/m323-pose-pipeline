"""Visualisierung: Label vs. YOLO-Vorhersage vs. PCK-Vergleich, pro Bild drei Panels.

    uv run python visualize.py [--split val|train|all] [--conf 0.25] [--alpha 0.1]

Schreibt out/vis_<datei>.png pro Bild und out/vis_<split>.png als Übersicht.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from pipeline.data import load_or_build, row_to_person
from pipeline.draw import draw_skeleton
from pipeline.evaluate import ALPHA, _dist, image_counts, match, pck, person_counts
from pipeline.geometry import to_pixels
from pipeline.inference import confident, predict_split
from pipeline.types import Person

KP_CONF = 0.5  # nur zum Zeichnen; PCK zählt laut Anleitung alle Punkte
OUT = Path("out")
FONT = ImageFont.load_default(size=16)



def draw_box(image: Image.Image, p_px: Person, color, label: str = "") -> Image.Image:
    out = image.copy()
    d = ImageDraw.Draw(out)
    cx, cy, w, h = p_px.box
    d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=color, width=2)
    if label:
        d.text((cx - w / 2 + 3, cy - h / 2 + 2), label, fill=color, font=FONT)
    return out


def draw_persons(image: Image.Image, persons, color, labels) -> Image.Image:
    out = image.convert("RGB")
    for p, lab in zip(persons, labels):
        out = draw_box(draw_skeleton(out, p), p, color, lab)
    return out


def draw_comparison(image: Image.Image, gts, preds, alpha: float) -> Image.Image:
    """Pro markiertem Label-Punkt: Kreis mit Radius alpha*Grösse, Linie zur Vorhersage.
    Grün = richtig (innerhalb), Rot = falsch, Grau = keine passende Vorhersage."""
    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    for gt in gts:
        pred = match(gt, preds)
        r = alpha * max(gt.box[2], gt.box[3])
        for i, g in enumerate(gt.keypoints):
            if g.v <= 0:
                continue
            if pred is None:
                d.ellipse([g.x - 4, g.y - 4, g.x + 4, g.y + 4], outline=(150, 150, 150), width=2)
                continue
            q = pred.keypoints[i]
            ok = _dist(g, q) <= r
            c = (0, 200, 0) if ok else (230, 0, 0)
            d.ellipse([g.x - r, g.y - r, g.x + r, g.y + r], outline=c, width=1)
            d.line([(g.x, g.y), (q.x, q.y)], fill=c, width=2)
            d.ellipse([g.x - 3, g.y - 3, g.x + 3, g.y + 3], fill=(255, 255, 255))
    return out


def header(image: Image.Image, text: str) -> Image.Image:
    bar = 26
    out = Image.new("RGB", (image.width, image.height + bar), (30, 30, 30))
    out.paste(image, (0, bar))
    ImageDraw.Draw(out).text((6, 4), text, fill=(255, 255, 255), font=FONT)
    return out


def hstack(images) -> Image.Image:
    h = max(i.height for i in images)
    out = Image.new("RGB", (sum(i.width for i in images) + 8 * (len(images) - 1), h), (255, 255, 255))
    x = 0
    for i in images:
        out.paste(i, (x, 0))
        x += i.width + 8
    return out


def vstack(images, width: int) -> Image.Image:
    scaled = [i.resize((width, round(i.height * width / i.width))) for i in images]
    out = Image.new("RGB", (width, sum(i.height for i in scaled) + 8 * (len(scaled) - 1)), (255, 255, 255))
    y = 0
    for i in scaled:
        out.paste(i, (0, y))
        y += i.height + 8
    return out


def panels(row: dict, alpha: float, kp_conf: float) -> Image.Image:
    w, h = row["width"], row["height"]
    gts = tuple(to_pixels(row_to_person(d), w, h) for d in row["persons"])
    preds = tuple(to_pixels(row_to_person(d), w, h) for d in row["pred"])
    c = image_counts(row, alpha)
    score = pck(c)
    img = row["image"]

    left = draw_persons(img, gts, (255, 220, 0), [f"L{i}" for i in range(len(gts))])
    mid = draw_persons(
        img, tuple(confident(p, kp_conf) for p in preds), (0, 220, 255),
        [f"{p.score:.2f}" for p in preds],
    )
    right = draw_comparison(img, gts, preds, alpha)
    return hstack([
        header(left, f"Label: {len(gts)} Personen"),
        header(mid, f"YOLO: {len(preds)} Personen (Punkte >= {kp_conf})"),
        header(right, f"PCK@{alpha}: {c.correct}/{c.total} = "
                      + ("-" if score is None else f"{score:.3f}")),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--kp-conf", type=float, default=KP_CONF)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    dsd = load_or_build()
    splits = ["train", "val"] if args.split == "all" else [args.split]
    for split in splits:
        ds_pred = predict_split(dsd[split], conf=args.conf)
        rows = [panels(row, args.alpha, args.kp_conf) for row in ds_pred]
        for row, im in zip(ds_pred, rows):
            im.save(OUT / f"vis_{Path(row['file_name']).stem}.png")
        vstack(rows, 1500).save(OUT / f"vis_{split}.png")
        print(f"{split}: {len(rows)} Bilder -> {OUT}/vis_{split}.png")


if __name__ == "__main__":
    main()
