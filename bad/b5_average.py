# Runde 2 – Schritt 5: Fold und Monoid.
# Startet mit `uv run python -m bad.b5_average`.
# Beobachtet:
#   - Was ergeben reduce(avg2, [1.0, 0.5, 0.0]) und die umgekehrte Reihenfolge?
#   - Weichen die gemittelten PCK-Werte vom Gesamt-PCK (add_counts) ab? Warum?
#   - Welche Ergebnisse ändern sich bei den Teil-Reihenfolgen (shard), welche nicht?

from functools import partial, reduce
from itertools import permutations

from pipeline.data import load_or_build
from pipeline.evaluate import ALPHA, ZERO, add_counts, image_counts, pck
from pipeline.inference import predict_split


def avg2(a: float, b: float) -> float:
    return (a + b) / 2


if __name__ == "__main__":
    print("--- 1. Vorgegebene Zahlen ---")
    print("reduce(avg2, [1.0, 0.5, 0.0]) =", reduce(avg2, [1.0, 0.5, 0.0]))
    print("reduce(avg2, [0.0, 0.5, 1.0]) =", reduce(avg2, [0.0, 0.5, 1.0]))

    ds_pred = predict_split(load_or_build()["val"])
    score = partial(image_counts, alpha=ALPHA)
    counts = [score(r) for r in ds_pred]
    pcks = [pck(c) for c in counts]

    print("\n--- 2. Echte PCK-Werte pro Bild ---")
    print("Counts:", counts)
    print("PCK   :", [round(p, 3) for p in pcks])
    print(f"avg2 vorwärts : {reduce(avg2, pcks):.4f}")
    print(f"avg2 rückwärts: {reduce(avg2, pcks[::-1]):.4f}")
    print(f"Mittel aller PCK-Werte : {sum(pcks) / len(pcks):.4f}")
    print(f"add_counts (korrekt)   : {pck(reduce(add_counts, counts, ZERO)):.4f}")

    print("\n--- 3. Drei Teile mit ds_pred.shard(3, i) ---")
    parts = [ds_pred.shard(3, i) for i in range(3)]
    print("Grösse der Teile:", [len(p) for p in parts])
    part_counts = [reduce(add_counts, map(score, p), ZERO) for p in parts]
    part_pcks = [pck(c) for c in part_counts]
    print("Counts pro Teil:", part_counts)
    print("PCK pro Teil   :", [round(p, 3) for p in part_pcks])
    print(f"Mittelwert der drei Teil-PCK: {sum(part_pcks) / 3:.4f}")
    for order in permutations(range(3)):
        add = reduce(add_counts, [part_counts[i] for i in order], ZERO)
        avg = reduce(avg2, [part_pcks[i] for i in order])
        print(f"Reihenfolge {order}: add_counts -> {pck(add):.4f} | avg2 -> {avg:.4f}")
