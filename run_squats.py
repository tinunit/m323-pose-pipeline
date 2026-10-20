"""Kniebeugen in einem Video zählen.

Aufruf: uv run python run_squats.py video.mp4 [--side left|right] [--plot]
"""

import argparse
from pathlib import Path

from pipeline.inference import WEIGHTS
from pipeline.reps import (  # noqa: F401
    DOWN_BELOW,
    UP_ABOVE,
    angles,
    count_reps,
    drop_missing,
    ema,
    frames,
)

ALPHA = 0.5  # Glättung, ausprobieren (siehe Aufgabe 9.2 c)


def plot(raw: list, smooth: list[float], out: str = "out/angles.png") -> None:  # vorgegeben
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(raw)), [float("nan") if a is None else a for a in raw], label="roh", alpha=0.5)
    ax.plot(range(len(smooth)), smooth, label=f"geglättet (alpha={ALPHA})")
    ax.axhline(DOWN_BELOW, color="red", linestyle="--", label=f"down < {DOWN_BELOW}")
    ax.axhline(UP_ABOVE, color="green", linestyle="--", label=f"up > {UP_ABOVE}")
    ax.set_xlabel("Bild")
    ax.set_ylabel("Kniewinkel (Grad)")
    ax.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, bbox_inches="tight")
    print("Diagramm:", out)


def main() -> None:
    ap = argparse.ArgumentParser()  # vorgegeben
    ap.add_argument("video")
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    # TODO 1. Winkel pro Bild: raw = list(angles(frames(args.video), WEIGHTS, args.side))
    # TODO 2. Fehlende Werte weglassen und glätten: smooth = ema(drop_missing(raw), ALPHA)
    # TODO 3. Ausgeben: Anzahl Bilder, Anzahl Bilder ohne Winkel (raw.count(None))
    #         und die Anzahl Kniebeugen: count_reps(smooth)
    # TODO 4. Mit --plot das Diagramm zeichnen: plot(raw, smooth)
    raise NotImplementedError("run_squats.py: TODO-Liste in main() abarbeiten")


if __name__ == "__main__":
    main()
