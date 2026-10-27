"""Erzeugt die Abbildungen für Kapitel 1 (YOLO und Pose).

    uv run python -m docs.make_figures

Alle Bilder stammen aus COCO8-pose (CC BY 4.0), die Ausgaben aus den YOLO11n-Modellen.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from pipeline.data import parse_label_file
from pipeline.draw import draw_skeleton
from pipeline.geometry import angle_at, to_pixels
from pipeline.inference import confident, predict_image
from pipeline.types import Person

IMG = Path("data/coco8-pose/images")
LBL = Path("data/coco8-pose/labels")
OUT = Path("docs/img")
def _font(size: int):
    """DejaVu kann Umlaute; die eingebaute Pillow-Schrift nicht immer."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


FONT = _font(20)
SMALL = _font(15)

NAMES = (
    "Nase", "Auge L", "Auge R", "Ohr L", "Ohr R", "Schulter L", "Schulter R",
    "Ellbogen L", "Ellbogen R", "Handgelenk L", "Handgelenk R", "Hüfte L", "Hüfte R",
    "Knie L", "Knie R", "Fussgelenk L", "Fussgelenk R",
)


def header(im: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (im.width, im.height + 32), (30, 30, 30))
    out.paste(im, (0, 32))
    ImageDraw.Draw(out).text((8, 5), text, fill=(255, 255, 255), font=FONT)
    return out


def grid(images, cols: int, cell_w: int) -> Image.Image:
    scaled = [i.resize((cell_w, round(i.height * cell_w / i.width))) for i in images]
    rows = [scaled[i:i + cols] for i in range(0, len(scaled), cols)]
    heights = [max(i.height for i in r) for r in rows]
    out = Image.new("RGB", (cols * cell_w + (cols - 1) * 8, sum(heights) + 8 * (len(rows) - 1)), "white")
    y = 0
    for r, h in zip(rows, heights):
        for c, im in enumerate(r):
            out.paste(im, (c * (cell_w + 8), y))
        y += h + 8
    return out


def plotted(model: str, path: Path) -> Image.Image:
    r = YOLO(model).predict(str(path), verbose=False, device="cpu")[0]
    return Image.fromarray(r.plot()[..., ::-1])  # BGR -> RGB


def tasks_figure(path: Path) -> None:
    """Klassifikation, Detektion, Segmentierung, Pose auf demselben Bild."""
    original = Image.open(path).convert("RGB")
    cls = YOLO("yolo11n-cls.pt").predict(str(path), verbose=False, device="cpu")[0]
    top = cls.probs.top5[:3]
    cls_im = original.copy()
    d = ImageDraw.Draw(cls_im)
    for i, idx in enumerate(top):
        d.text((10, 10 + 26 * i), f"{cls.names[idx]}: {cls.probs.data[idx]:.2f}",
               fill="yellow", font=FONT, stroke_width=3, stroke_fill="black")
    panels = [
        header(cls_im, "Klassifikation: ganzes Bild -> Kategorie"),
        header(plotted("yolo11n.pt", path), "Detektion: Boxen + Klasse"),
        header(plotted("yolo11n-seg.pt", path), "Segmentierung: Pixelmaske"),
        header(plotted("yolo11n-pose.pt", path), "Pose: Boxen + 17 Keypoints"),
    ]
    grid(panels, 2, 640).save(OUT / "yolo_aufgaben.png")


def load(split: str, stem: str, person: int) -> tuple[Image.Image, Person]:
    """Bild und eine Label-Person in Pixeln."""
    im = Image.open(IMG / split / f"{stem}.jpg").convert("RGB")
    p = parse_label_file((LBL / split / f"{stem}.txt").read_text())[person]
    return im, to_pixels(p, im.width, im.height)


def crop_box(p_px: Person, im: Image.Image, margin: float = 0.15) -> tuple[int, int, int, int]:
    cx, cy, w, h = p_px.box
    mx, my = w * margin, h * margin
    return (max(0, int(cx - w / 2 - mx)), max(0, int(cy - h / 2 - my)),
            min(im.width, int(cx + w / 2 + mx)), min(im.height, int(cy + h / 2 + my)))


def keypoints_figure(split: str, stem: str, person: int, scale: int = 2) -> None:
    """Eine Person, die Label-Keypoints nummeriert und benannt."""
    im, p_px = load(split, stem, person)
    x0, y0, x1, y1 = crop_box(p_px, im, margin=0.35)
    out = draw_skeleton(im, p_px).crop((x0, y0, x1, y1))
    out = out.resize((out.width * scale, out.height * scale))
    d = ImageDraw.Draw(out)
    for i, k in enumerate(p_px.keypoints):
        if k.v > 0:
            x, y = (k.x - x0) * scale, (k.y - y0) * scale
            right_side = i % 2 == 0 and i > 0  # rechte Körperseite: Text links vom Punkt
            text = f"{i} {NAMES[i]}"
            w = d.textlength(text, font=SMALL)
            tx = x - 12 - w if right_side else x + 12
            tx = max(4, min(tx, out.width - w - 4))  # nicht über den Bildrand
            dy = -26 if i in (3, 4) else -9  # Ohren über den Punkt, sonst überlappen sie mit den Augen
            d.text((tx, y + dy), text, fill="white", font=SMALL, stroke_width=3, stroke_fill="black")
    out.save(OUT / "keypoints_nummeriert.png")


def label_vs_pred_figure(split: str, stem: str, person: int) -> None:
    """Links Label (von Menschen markiert), rechts Vorhersage von YOLO, gleicher Ausschnitt."""
    im, gt = load(split, stem, person)
    preds = tuple(to_pixels(confident(p), im.width, im.height) for p in predict_image(im))
    pred = min(preds, key=lambda q: abs(q.box[0] - gt.box[0]) + abs(q.box[1] - gt.box[1]))
    box = crop_box(gt, im)
    left = draw_skeleton(im, gt).crop(box)
    right = draw_skeleton(im, pred).crop(box)
    grid([header(left, "Label (Mensch)"), header(right, f"YOLO ({pred.score:.2f})")],
         2, 420).save(OUT / "label_vs_yolo.png")


def _axes_panel(im: Image.Image, p_draw: Person, ticks_x, ticks_y, fmt, kp: int,
                kp_text: str, angle_text: str, title: str) -> Image.Image:
    """Bild mit Achsen oben/links (Ursprung oben links, y nach unten), Box und einem markierten Punkt."""
    pad_l, pad_t, pad_b = 70, 70, 70
    W, H = im.width, im.height
    out = Image.new("RGB", (W + pad_l + 20, H + pad_t + pad_b), "white")
    out.paste(draw_skeleton(im, p_draw), (pad_l, pad_t))
    d = ImageDraw.Draw(out)
    ox, oy = pad_l, pad_t
    black = (0, 0, 0)
    # Achsen
    d.line([(ox, oy), (ox + W + 12, oy)], fill=black, width=2)
    d.polygon([(ox + W + 14, oy), (ox + W + 4, oy - 5), (ox + W + 4, oy + 5)], fill=black)
    d.line([(ox, oy), (ox, oy + H + 12)], fill=black, width=2)
    d.polygon([(ox, oy + H + 14), (ox - 5, oy + H + 4), (ox + 5, oy + H + 4)], fill=black)
    d.text((ox + W - 10, oy - 52), "x", fill=black, font=FONT)
    d.text((ox - 60, oy + H + 2), "y", fill=black, font=FONT)
    for value, pos in ticks_x:
        d.line([(ox + pos, oy - 6), (ox + pos, oy)], fill=black, width=2)
        t = fmt(value)
        d.text((ox + pos - d.textlength(t, font=SMALL) / 2, oy - 26), t, fill=black, font=SMALL)
    for value, pos in ticks_y:
        d.line([(ox - 6, oy + pos), (ox, oy + pos)], fill=black, width=2)
        t = fmt(value)
        d.text((ox - 10 - d.textlength(t, font=SMALL), oy + pos - 9), t, fill=black, font=SMALL)
    d.text((ox + 4, oy + 4), "(0, 0)", fill="white", font=SMALL, stroke_width=3, stroke_fill="black")
    # Box
    cx, cy, w, h = p_draw.box
    d.rectangle([ox + cx - w / 2, oy + cy - h / 2, ox + cx + w / 2, oy + cy + h / 2], outline=(255, 200, 0), width=3)
    d.ellipse([ox + cx - 5, oy + cy - 5, ox + cx + 5, oy + cy + 5], fill=(255, 200, 0), outline=black)
    d.text((ox + cx + 8, oy + cy - 10), "(cx, cy)", fill=(255, 200, 0), font=SMALL, stroke_width=3, stroke_fill="black")
    # markierter Punkt mit Hilfslinien zu den Achsen
    k = p_draw.keypoints[kp]
    px, py = ox + k.x, oy + k.y
    for i in range(0, int(k.x), 12):
        d.line([(ox + i, py), (ox + min(i + 6, k.x), py)], fill=(255, 0, 255), width=2)
    for i in range(0, int(k.y), 12):
        d.line([(px, oy + i), (px, oy + min(i + 6, k.y))], fill=(255, 0, 255), width=2)
    d.ellipse([px - 8, py - 8, px + 8, py + 8], outline=(255, 0, 255), width=4)
    d.text((px + 12, py - 30), kp_text, fill="white", font=SMALL, stroke_width=3, stroke_fill="black")
    d.text((8, oy + H + 24), angle_text, fill=black, font=FONT)
    return header(out, title)


def coordinates_figure(split: str, stem: str, person: int, joint=(6, 8, 10), joint_name: str = "Ellbogenwinkel R") -> None:
    """Links Pixel (Bild wie es ist), rechts normalisiert (Bild auf 0..1 x 0..1 gequetscht)."""
    im = Image.open(IMG / split / f"{stem}.jpg").convert("RGB")
    p = parse_label_file((LBL / split / f"{stem}.txt").read_text())[person]
    W, H = im.size
    p_px = to_pixels(p, W, H)
    kp = joint[1]
    k, k_px = p.keypoints[kp], p_px.keypoints[kp]
    ang_px = angle_at(*(p_px.keypoints[i] for i in joint))
    ang_n = angle_at(*(p.keypoints[i] for i in joint))

    left = _axes_panel(
        im, p_px,
        [(v, v) for v in range(0, W + 1, 100)], [(v, v) for v in range(0, H + 1, 100)],
        lambda v: f"{v}", kp,
        f"{NAMES[kp]}: ({k_px.x:.0f}, {k_px.y:.0f}) px",
        f"{joint_name} in Pixeln: {ang_px:.1f}°",
        f"Pixel: Bild {W} × {H}",
    )
    S = 500  # Quadrat: so "sieht" die normalisierte Welt das Bild
    sq = im.resize((S, S))
    p_sq = to_pixels(p, S, S)
    ticks = [(v, v * S) for v in (0, 0.25, 0.5, 0.75, 1.0)]
    right = _axes_panel(
        sq, p_sq, ticks, ticks, lambda v: f"{v:g}", kp,
        f"{NAMES[kp]}: ({k.x:.3f}, {k.y:.3f})",
        f"{joint_name} normalisiert: {ang_n:.1f}° (verzerrt)",
        "Normalisiert: alles 0 bis 1",
    )
    grid([left, right], 2, 620).save(OUT / "koordinaten.png")
    print(f"{NAMES[kp]}: norm ({k.x:.4f}, {k.y:.4f}) px ({k_px.x:.1f}, {k_px.y:.1f})  "
          f"Winkel px {ang_px:.1f}  norm {ang_n:.1f}  Bild {W}x{H}")


def first_steps_figure() -> None:
    """Die drei Beispiele aus "YOLO in 5 Zeilen": genau das, was result.save(...) liefert."""
    panels = [
        header(plotted("yolo11n-cls.pt", IMG / "train" / "000000000036.jpg"), "Beispiel 1: Klassifikation"),
        header(plotted("yolo11n-cls.pt", IMG / "val" / "000000000110.jpg"), "Beispiel 2: Klassifikation"),
        header(plotted("yolo11n-pose.pt", IMG / "val" / "000000000113.jpg"), "Beispiel 3: Pose"),
    ]
    h = 520
    scaled = [p.resize((round(p.width * h / p.height), h)) for p in panels]
    out = Image.new("RGB", (sum(p.width for p in scaled) + 8 * (len(scaled) - 1), h), "white")
    x = 0
    for p in scaled:
        out.paste(p, (x, 0))
        x += p.width + 8
    out.save(OUT / "erste_schritte.jpg", quality=85)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    first_steps_figure()
    coordinates_figure("val", "000000000113", person=0)
    tasks_figure(IMG / "val" / "000000000113.jpg")
    keypoints_figure("val", "000000000113", person=0)
    label_vs_pred_figure("val", "000000000113", person=0)
    print("\n".join(str(p) for p in sorted(OUT.glob("*.png"))))
