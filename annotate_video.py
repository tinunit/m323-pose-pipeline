"""Video mit Skelett, Kniewinkel und Zähler pro Bild.

    uv run python annotate_video.py videos/video.mp4 [--side left] [--down 100] [--up 160] [--out out/video/annotated.mp4]

Zum Prüfen von Schritt 6: Man sieht für jedes Bild, welchen Winkel die Pipeline
misst, in welchem Zustand der Zähler ist und wann gezählt wird.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.draw import draw_skeleton
from pipeline.geometry import KNEE_JOINTS, knee_angle, to_pixels
from pipeline.inference import confident, predict_image
from pipeline.reps import START, RepState, frames


def make_step(down: float, up: float):
    """Wie reps.step, aber mit wählbaren Schwellen (zum Ausprobieren)."""
    def step(state: RepState, angle: float) -> RepState:
        if state.phase == "up" and angle < down:
            return RepState("down", state.count)
        if state.phase == "down" and angle > up:
            return RepState("up", state.count + 1)
        return state
    return step


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--side", default="left", choices=["left", "right"])
    ap.add_argument("--down", type=float, default=100)
    ap.add_argument("--up", type=float, default=160)
    ap.add_argument("--width", type=int, default=960, help="Breite des Ausgabevideos")
    ap.add_argument("--out", default="out/video/annotated.mp4")
    args = ap.parse_args()

    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS) or 30
    step = make_step(args.down, args.up)
    state, writer = START, None
    big, small = font(34), font(22)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames(args.video)):
        img = Image.fromarray(frame)
        h, w = frame.shape[:2]
        persons = predict_image(img)
        angle = None
        if persons:
            p_px = to_pixels(confident(persons[0]), w, h)
            img = draw_skeleton(img, p_px)
            angle = knee_angle(p_px, args.side)
            if angle is not None:
                state = step(state, angle)
                knee = p_px.keypoints[KNEE_JOINTS[args.side][1]]
                d = ImageDraw.Draw(img)
                d.ellipse([knee.x - 18, knee.y - 18, knee.x + 18, knee.y + 18], outline="yellow", width=5)
                d.text((knee.x + 26, knee.y - 20), f"{angle:.0f}°", fill="yellow", font=big,
                       stroke_width=4, stroke_fill="black")

        scale = args.width / w
        img = img.resize((args.width, round(h * scale)))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, img.width, 92], fill=(0, 0, 0))
        knee_txt = "–" if angle is None else f"{angle:.0f}°"
        color = (255, 120, 120) if state.phase == "down" else (120, 255, 120)
        d.text((12, 8), f"Wiederholungen: {state.count}", fill="white", font=big)
        d.text((12, 56), f"Bild {i}   Knie {args.side}: {knee_txt}   Zustand: ", fill="white", font=small)
        x = 12 + d.textlength(f"Bild {i}   Knie {args.side}: {knee_txt}   Zustand: ", font=small)
        d.text((x, 56), state.phase, fill=color, font=small)
        d.text((img.width - 260, 56), f"down < {args.down:.0f}°, up > {args.up:.0f}°", fill="white", font=small)

        out = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.width, img.height))
        writer.write(out)
    if writer is not None:
        writer.release()
    print(f"Wiederholungen: {state.count}  ->  {args.out}")


if __name__ == "__main__":
    main()
