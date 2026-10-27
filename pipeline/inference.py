"""Schritt 3: Inferenz mit YOLO.

Pure core (reiner Kern): result_to_persons, confident (nur Zahlen rein, Personen raus).
Impure shell (unreine Schale): get_model, predict_batch, predict_image (Modell, Dateien, CPU).
"""

import sys
from functools import lru_cache
from typing import Optional

from datasets import Dataset, Features
from PIL import Image
from ultralytics import YOLO

from pipeline.data import FEATURES, PERSON_FEATURE, person_to_row
from pipeline.types import Keypoint, Person

# --- vorgegeben ---------------------------------------------------------------
# Wie FEATURES, plus Spalte "pred" mit demselben Aufbau wie "persons".
FEATURES_WITH_PRED = Features({**FEATURES, "pred": [PERSON_FEATURE]})

WEIGHTS = "yolo11n-pose.pt"
KP_CONF = 0.5  # vorhergesagte Punkte darunter gelten als "nicht markiert"


# --- Runde 1: eure Aufgabe ----------------------------------------------------
@lru_cache(maxsize=None)
def get_model(weights: str) -> YOLO:
    """Impure. Lädt das YOLO-Modell. Dank lru_cache nur beim ersten Aufruf."""
    # TODO: das Modell mit YOLO(weights) laden und zurückgeben (eine Zeile, wie in "Los geht's")
    raise NotImplementedError("Schritt 3: get_model")


def result_to_persons(
    boxes_xywhn: list[list[float]],
    boxes_conf: list[float],
    kps_xyn: list[list[list[float]]],
    kps_conf: Optional[list[list[float]]],
) -> tuple[Person, ...]:
    """Pure. Zahlenlisten aus dem YOLO-Ergebnis in Personen (keine YOLO-Objekte, darum ohne Modell testbar).
    Keine Erkennung ergibt ()."""
    if kps_conf is None:                                     # gegeben: ohne Konfidenz gilt v = 1.0
        kps_conf = [[1.0] * len(xy) for xy in kps_xyn]
    persons = []
    for box, score, xy, conf in zip(boxes_xywhn, boxes_conf, kps_xyn, kps_conf):
        # TODO: kps soll ein Tupel von 17 Keypoint(x, y, v) sein.
        #   xy   = [[x0, y0], [x1, y1], ...]   (17 Paare, normalisiert)
        #   conf = [v0, v1, ...]               (17 Konfidenzen)
        #   Ergebnis z.B. (Keypoint(0.52, 0.38, 0.97), Keypoint(0.55, 0.35, 0.95), ...)
        kps = ...
        raise NotImplementedError("Schritt 3: kps ergänzen, dann diese Zeile löschen")
        persons.append(Person(box=tuple(box), keypoints=kps, score=score))
    return tuple(persons)


# --- vorgegeben ---------------------------------------------------------------
def confident(p: Person, kp_conf: float = KP_CONF) -> Person:
    """Pure. Punkte mit Konfidenz < kp_conf bekommen v = 0.

    Bei Vorhersagen ist v die Konfidenz und praktisch nie genau 0. Ohne diesen
    Schritt gilt jeder geratene Punkt (z.B. ein Knie ausserhalb des Bildes) als
    markiert, und knee_angle liefert einen Winkel statt None.
    """
    return p._replace(keypoints=tuple(
        k if k.v >= kp_conf else Keypoint(k.x, k.y, 0.0) for k in p.keypoints
    ))


def _persons_of(result) -> tuple[Person, ...]:
    """Dünne Schale: YOLO-Ergebnis -> Zahlenlisten -> Personen."""
    kps = result.keypoints
    return result_to_persons(
        result.boxes.xywhn.tolist(),
        result.boxes.conf.tolist(),
        kps.xyn.tolist(),
        kps.conf.tolist() if kps.conf is not None else None,
    )


def predict_batch(batch: dict, weights: str, conf: float) -> dict:
    """Impure. Für Dataset.map(batched=True). Gibt {"pred": [...]} zurück."""
    images = [im.convert("RGB") for im in batch["image"]]
    results = get_model(weights).predict(images, conf=conf, verbose=False, device="cpu")
    return {"pred": [[person_to_row(p) for p in _persons_of(r)] for r in results]}


def predict_split(ds: Dataset, weights: str = WEIGHTS, conf: float = 0.25) -> Dataset:
    """ds.map über das ganze Dataset. Nur bei einem Dataset von der Festplatte
    (load_from_disk) wird das Ergebnis zwischengespeichert."""
    return ds.map(
        predict_batch,
        batched=True,
        batch_size=4,
        fn_kwargs={"weights": weights, "conf": conf},
        features=FEATURES_WITH_PRED,
    )


def predict_image(image: Image.Image, weights: str = WEIGHTS, conf: float = 0.25) -> tuple[Person, ...]:
    """Impure. ein einzelnes Bild (PIL; numpy-Arrays gehen über Image.fromarray)."""
    result = get_model(weights).predict(
        [image.convert("RGB")], conf=conf, verbose=False, device="cpu"
    )[0]
    return _persons_of(result)


def annotate(path: str, weights: str = WEIGHTS, conf: float = 0.25, out: str = "out/annotated.png") -> None:
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


def demo_cache(conf: float = 0.25, pass_model: bool = False) -> None:
    """Die Beobachtungen aus Abschnitt 3.3: map-Cache, conf ändern, Modellobjekt als Argument.

    Aufruf: python -m pipeline.inference --demo-cache [conf] [--model-object]
    """
    import datasets
    from pipeline.data import load_or_build

    datasets.logging.set_verbosity_info()
    val = load_or_build()["val"]
    if pass_model:
        ds_pred = val.map(
            lambda b, model: {
                "pred": [
                    [person_to_row(p) for p in _persons_of(r)]
                    for r in model.predict(
                        [im.convert("RGB") for im in b["image"]],
                        conf=conf, verbose=False, device="cpu",
                    )
                ]
            },
            batched=True,
            batch_size=4,
            fn_kwargs={"model": get_model(WEIGHTS)},
            features=FEATURES_WITH_PRED,
        )
    else:
        ds_pred = predict_split(val, WEIGHTS, conf)
    print("Personen pro Bild:", [len(r) for r in ds_pred["pred"]])


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--demo-cache":
        rest = [a for a in args[1:] if not a.startswith("--")]
        demo_cache(float(rest[0]) if rest else 0.25, "--model-object" in args)
    elif args:
        annotate(args[0])
    else:
        print("Aufruf: python -m pipeline.inference bild.jpg  |  --demo-cache [conf] [--model-object]")
