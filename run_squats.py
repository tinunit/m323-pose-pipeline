"""Kniebeugen in einem Video zählen.

Aufruf: uv run python run_squats.py videos/kniebeugen.mp4 [--side left|right] [--plot]
"""

import argparse
from pathlib import Path

from pipeline.reps import (
    DOWN_BELOW,
    UP_ABOVE,
    angles,
    count_reps,
    drop_missing,
    ema,
    frames,
)
from pipeline.inference import WEIGHTS

ALPHA = 0.5


def plot(raw: list, smooth: list[float], out: str = "out/angles.png") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(raw)), [float("nan") if a is None else a for a in raw], label="roh", alpha=0.5)
    ax.plot([i for i, a in enumerate(raw) if a is not None], smooth, label=f"geglättet (alpha={ALPHA})")
    ax.axhline(DOWN_BELOW, color="red", linestyle="--", label=f"down < {DOWN_BELOW}")
    ax.axhline(UP_ABOVE, color="green", linestyle="--", label=f"up > {UP_ABOVE}")
    ax.set_xlabel("Bild")
    ax.set_ylabel("Kniewinkel (Grad)")
    ax.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print("Diagramm:", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    raw = list(angles(frames(args.video), WEIGHTS, args.side))  # Bild für Bild (lazy)
    smooth = ema(drop_missing(raw), ALPHA)
    print(f"Bilder: {len(raw)}, ohne Person/Winkel: {raw.count(None)}")
    print(f"Kniebeugen: {count_reps(smooth)}")
    if args.plot:
        plot(raw, smooth)


if __name__ == "__main__":
    main()
