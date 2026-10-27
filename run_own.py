"""Teil C (Schritt 3): eigene Bilder durch die Pose-Pipeline.

    uv run python run_own.py                 # alle Bilder in eigene/
    uv run python run_own.py foto1.jpg ...   # einzelne Dateien
    uv run python run_own.py --dir ~/Pictures/m323-pose   # anderer Ordner -> <ordner>/ergebnisse/
    uv run python run_own.py --conf 0.5 --max-side 1280

Ablauf: Bild normalisieren (EXIF-Drehung, RGB, verkleinern) -> Dataset im selben
Schema wie COCO8 (ohne Labels) -> predict_split (map) -> to_pixels -> Kniewinkel
-> Skelett zeichnen. Ausgabe: out/eigene/<name>.png und eine Tabelle.
Eigene Bilder nicht ins Repo committen (eigene/ steht in .gitignore).
"""

import argparse
from pathlib import Path

from datasets import Dataset
from PIL import Image, ImageDraw, ImageFont, ImageOps

from pipeline.data import FEATURES, row_to_person
from pipeline.draw import draw_skeleton
from pipeline.geometry import knee_angle, to_pixels
from pipeline.inference import KP_CONF, confident, predict_split

IN_DIR = Path("eigene")
OUT_DIR = Path("out/eigene")
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIDE = 1280
FONT = ImageFont.load_default(size=18)


def normalize_image(im: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    """Pure. neues Bild, aufrecht (EXIF-Orientierung angewendet), RGB, längste Seite <= max_side.

    Handyfotos speichern die Drehung oft nur als EXIF-Tag. Ohne exif_transpose
    sieht YOLO die Person liegend.
    """
    out = ImageOps.exif_transpose(im).convert("RGB")
    scale = max_side / max(out.size)
    if scale < 1:
        out = out.resize((round(out.width * scale), round(out.height * scale)), Image.LANCZOS)
    return out


def image_paths(args: list[str], in_dir: Path = IN_DIR) -> list[Path]:
    """Impure. Dateien aus den Argumenten oder alle Bilder im Ordner in_dir."""
    paths = [Path(a) for a in args] if args else sorted(in_dir.expanduser().glob("*"))
    return [p for p in paths if p.suffix.lower() in SUFFIXES]


def to_row(name: str, im: Image.Image) -> dict:
    """Pure. ein Beispiel im Dataset-Schema, ohne Labels (persons = [])."""
    return {"file_name": name, "image": im, "width": im.width, "height": im.height, "persons": []}


def fmt(a) -> str:
    return "-" if a is None else f"{a:.1f}"


def annotate_row(row: dict, kp_conf: float = KP_CONF) -> tuple[Image.Image, list[tuple[float, float | None, float | None]]]:
    """Pure. Skelette aller erkannten Personen + Kniewinkel. Gibt Bild und (score, links, rechts) zurück."""
    w, h = row["width"], row["height"]
    preds = tuple(confident(to_pixels(row_to_person(d), w, h), kp_conf) for d in row["pred"])
    out = row["image"].convert("RGB")
    infos = []
    for i, p in enumerate(preds):
        out = draw_skeleton(out, p)
        left, right = knee_angle(p, "left"), knee_angle(p, "right")
        infos.append((p.score, left, right))
        cx, cy, bw, bh = p.box
        ImageDraw.Draw(out).text(
            (cx - bw / 2, max(0, cy - bh / 2 - 22)),
            f"#{i} {p.score:.2f}  L {fmt(left)}  R {fmt(right)}",
            fill=(255, 255, 0), font=FONT, stroke_width=2, stroke_fill=(0, 0, 0),
        )
    return out, infos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--dir", type=Path, default=IN_DIR, help="Ordner mit Bildern (Standard: eigene/)")
    ap.add_argument("--out", type=Path, default=None, help="Ausgabe (Standard: out/eigene/, bei --dir: <dir>/ergebnisse/)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-side", type=int, default=MAX_SIDE)
    ap.add_argument("--kp-conf", type=float, default=KP_CONF, help="Punkte darunter gelten als nicht markiert")
    args = ap.parse_args()

    paths = image_paths(args.files, args.dir)
    out_dir = args.out or (OUT_DIR if args.dir == IN_DIR else args.dir.expanduser() / "ergebnisse")
    if not paths:
        print(f"Keine Bilder gefunden. Legt Fotos in {args.dir}/ oder gebt Dateien an.")
        return

    rows = []
    for p in paths:
        with Image.open(p) as im:
            rows.append(to_row(p.name, normalize_image(im, args.max_side)))
    ds = Dataset.from_list(rows, features=FEATURES)  # im Speicher: kein map-Cache
    ds_pred = predict_split(ds, conf=args.conf)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'Datei':30} {'Grösse':>11}  Person  Score  Knie L  Knie R")
    for row in ds_pred:
        im, infos = annotate_row(row, args.kp_conf)
        im.save(out_dir / (Path(row["file_name"]).stem + ".png"))
        size = f"{row['width']}x{row['height']}"
        if not infos:
            print(f"{row['file_name']:30} {size:>11}  keine Person erkannt")
        for i, (score, left, right) in enumerate(infos):
            print(f"{row['file_name']:30} {size:>11}  #{i:<5} {score:5.2f}  {fmt(left):>6}  {fmt(right):>6}")
    print(f"\nErgebnisse: {out_dir}/")


if __name__ == "__main__":
    main()
