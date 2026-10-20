"""Skelett zeichnen (Pixelkoordinaten). Linke Seite blau, rechte Seite rot."""

from PIL import Image

from pipeline.types import Person

# --- vorgegeben ---------------------------------------------------------------
# COCO-Skelett, 0-basierte Indizes der Keypoints
SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)
LEFT = frozenset(range(1, 17, 2))   # 1, 3, ..., 15
RIGHT = frozenset(range(2, 17, 2))  # 2, 4, ..., 16
BLUE = (0, 80, 255)    # linke Seite
RED = (255, 40, 40)    # rechte Seite
GREEN = (0, 200, 0)    # Mitte / Verbindung zwischen den Seiten


# --- eure Aufgabe -------------------------------------------------------------
def draw_skeleton(image: Image.Image, p_px: Person) -> Image.Image:
    """Neues Bild mit Skelett (PIL, ImageDraw). Das Eingabebild bleibt unverändert.

    Für jede Kante (i, j) in SKELETON, bei der beide Keypoints markiert sind (v > 0),
    eine Linie zeichnen: blau, wenn beide Punkte links sind, rot, wenn beide rechts
    sind, sonst grün. Dazu je einen kleinen Kreis pro markiertem Keypoint
    (linke Seite blau, rechte Seite rot, Nase grün).
    Tipp: image.convert("RGB") kann dasselbe Objekt zurückgeben, mit image.copy() arbeiten.
    """
    raise NotImplementedError
