"""Skelett zeichnen (Pixelkoordinaten). Linke Seite blau, rechte Seite rot."""

from PIL import Image, ImageDraw

from pipeline.types import Person

# COCO-Skelett, 0-basierte Indizes
EDGES = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)
LEFT = frozenset(range(1, 17, 2))   # 1, 3, ..., 15
RIGHT = frozenset(range(2, 17, 2))  # 2, 4, ..., 16
BLUE, RED, GREEN = (0, 80, 255), (255, 40, 40), (0, 200, 0)


def _color(i: int, j: int) -> tuple[int, int, int]:
    if i in LEFT and j in LEFT:
        return BLUE
    if i in RIGHT and j in RIGHT:
        return RED
    return GREEN  # Mitte / Verbindung zwischen den Seiten


def draw_skeleton(image: Image.Image, p_px: Person) -> Image.Image:
    """Neues Bild mit Skelett. Das Eingabebild bleibt unverändert."""
    out = image.convert("RGB")  # liefert hier immer eine Kopie
    if out is image:
        out = image.copy()
    draw = ImageDraw.Draw(out)
    kps = p_px.keypoints
    for i, j in EDGES:
        if kps[i].v > 0 and kps[j].v > 0:
            draw.line([(kps[i].x, kps[i].y), (kps[j].x, kps[j].y)], fill=_color(i, j), width=3)
    for i, k in enumerate(kps):
        if k.v > 0:
            c = BLUE if i in LEFT else RED if i in RIGHT else GREEN
            draw.ellipse([k.x - 4, k.y - 4, k.x + 4, k.y + 4], fill=c)
    return out
