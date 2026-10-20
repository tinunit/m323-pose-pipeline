"""Schritt 4: Inferenz mit YOLO.

Reiner Kern: result_to_persons (nur Zahlenlisten rein, Personen raus).
Unreine Schale: get_model, predict_batch, predict_image (Modell, Dateien, CPU).
"""

import sys
from typing import Optional

from datasets import Dataset, Features
from PIL import Image
from ultralytics import YOLO

from pipeline.data import FEATURES, PERSON_FEATURE, person_to_row
from pipeline.types import Person

# --- vorgegeben ---------------------------------------------------------------
# Wie FEATURES, plus Spalte "pred" mit demselben Aufbau wie "persons".
FEATURES_WITH_PRED = Features({**FEATURES, "pred": [PERSON_FEATURE]})

WEIGHTS = "yolo11n-pose.pt"


# --- eure Aufgabe -------------------------------------------------------------
def get_model(weights: str) -> YOLO:
    """Unrein. Lädt das Modell: YOLO(weights).

    Mit @lru_cache dekorieren, damit das Modell nur einmal geladen wird.
    """
    raise NotImplementedError


def result_to_persons(
    boxes_xywhn: list[list[float]],
    boxes_conf: list[float],
    kps_xyn: list[list[list[float]]],
    kps_conf: Optional[list[list[float]]],
) -> tuple[Person, ...]:
    """Rein. Zahlenlisten aus dem YOLO-Ergebnis in Personen umwandeln.

    Nimmt nur Listen von Zahlen, nicht das YOLO-Objekt (so ohne Modell testbar).
    Pro Erkennung: box = xywhn, score = conf, 17 Keypoints aus kps_xyn
    (Konfidenz v aus kps_conf). Keine Erkennung ergibt ().
    Fehlt die Konfidenz der Keypoints (None), wird v = 1.0 gesetzt.
    """
    raise NotImplementedError


def predict_batch(batch: dict, weights: str, conf: float) -> dict:
    """Unrein. Für Dataset.map(batched=True).

    Bilder mit .convert("RGB") umwandeln, dann
    get_model(weights).predict(images, conf=conf, verbose=False, device="cpu").
    Für jedes Ergebnis result_to_persons aufrufen
    (r.boxes.xywhn.tolist(), r.boxes.conf.tolist(), r.keypoints.xyn.tolist(),
    r.keypoints.conf, kann None sein) und mit person_to_row in Zeilen umwandeln.
    Rückgabe: {"pred": [...]} (eine Liste von Personen-dicts pro Bild).
    """
    raise NotImplementedError


# --- vorgegeben (Plumbing) -----------------------------------------------------
def _persons_of(result) -> tuple[Person, ...]:  # vorgegeben
    """Dünne Schale: YOLO-Ergebnis -> Zahlenlisten -> Personen."""
    kps = result.keypoints
    return result_to_persons(
        result.boxes.xywhn.tolist(),
        result.boxes.conf.tolist(),
        kps.xyn.tolist(),
        kps.conf.tolist() if kps.conf is not None else None,
    )


def predict_split(ds: Dataset, weights: str = WEIGHTS, conf: float = 0.25) -> Dataset:  # vorgegeben
    """ds.map über das ganze Dataset. Nur bei einem Dataset von der Festplatte
    (load_from_disk) wird das Ergebnis zwischengespeichert."""
    return ds.map(
        predict_batch,
        batched=True,
        batch_size=4,
        fn_kwargs={"weights": weights, "conf": conf},
        features=FEATURES_WITH_PRED,
    )


def predict_image(image: Image.Image, weights: str = WEIGHTS, conf: float = 0.25) -> tuple[Person, ...]:  # vorgegeben
    """Unrein: ein einzelnes Bild (PIL; numpy-Arrays gehen über Image.fromarray)."""
    result = get_model(weights).predict(
        [image.convert("RGB")], conf=conf, verbose=False, device="cpu"
    )[0]
    return _persons_of(result)


def annotate(path: str, weights: str = WEIGHTS, conf: float = 0.25, out: str = "out/annotated.png") -> None:  # vorgegeben
    """Teil C: Bild laden, Inferenz, Skelett zeichnen, Kniewinkel ausgeben."""
    from pathlib import Path

    from pipeline.draw import draw_skeleton
    from pipeline.geometry import knee_angle, to_pixels

    img = Image.open(path).convert("RGB")
    w, h = img.size
    for i, p in enumerate(predict_image(img, weights, conf)):
        p_px = to_pixels(p, w, h)
        img = draw_skeleton(img, p_px)
        print(
            f"person {i}: score={p.score:.2f} "
            f"left_knee={knee_angle(p_px, 'left')} right_knee={knee_angle(p_px, 'right')}"
        )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print("gespeichert:", out)


def demo_cache(conf: float = 0.25) -> None:  # vorgegeben
    """Beobachtungen aus 7.2: map-Meldungen anschauen, conf ändern.

    Aufruf: uv run python -m pipeline.inference --demo-cache [conf]
    Für Punkt 3 (Modellobjekt statt weights als fn_kwargs) baut ihr selbst eine Variante.
    """
    import datasets
    from pipeline.data import load_or_build

    datasets.logging.set_verbosity_info()
    val = load_or_build()["val"]
    ds_pred = predict_split(val, WEIGHTS, conf)
    print("Personen pro Bild:", [len(r) for r in ds_pred["pred"]])


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--demo-cache":
        demo_cache(float(args[1]) if len(args) > 1 else 0.25)
    elif args:
        annotate(args[0])
    else:
        print("Aufruf: python -m pipeline.inference bild.jpg  |  --demo-cache [conf]")
