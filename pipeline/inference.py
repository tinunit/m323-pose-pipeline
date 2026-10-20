"""Schritt 4: Inferenz mit YOLO.

Reiner Kern: result_to_persons (nur Zahlenlisten rein, Personen raus).
Unreine Schale: get_model, predict_batch, predict_image (Modell, Dateien, CPU).
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
    """Unrein. Lädt das YOLO-Modell zu den Gewichten `weights`.

    Das Laden dauert; das Modell soll pro Gewichtsdatei nur einmal geladen
    werden. Der Decorator `lru_cache` darüber sorgt dafür, ihr schreibt nur
    das Laden selbst.
    """
    raise NotImplementedError


def result_to_persons(
    boxes_xywhn: list[list[float]],
    boxes_conf: list[float],
    kps_xyn: list[list[list[float]]],
    kps_conf: Optional[list[list[float]]],
) -> tuple[Person, ...]:
    """Rein. Zahlenlisten aus dem YOLO-Ergebnis in Personen umwandeln.

    Alle Listen haben eine Zeile pro erkannter Person:
    - boxes_xywhn: je [cx, cy, w, h] (normalisiert)
    - boxes_conf: je ein Wert, wird zu `score` der Person
    - kps_xyn: je 17 Paare [x, y]
    - kps_conf: je 17 Konfidenzen (werden zu v) oder None

    Ist kps_conf None, wird v = 1.0 für alle Punkte gesetzt. Keine Erkennung
    (leere Listen) ergibt ().
    """
    raise NotImplementedError


def predict_batch(batch: dict, weights: str, conf: float) -> dict:
    """Unrein. Für Dataset.map(batched=True): Inferenz für einen Batch Bilder.

    `batch["image"]` ist eine Liste von PIL-Bildern. Mit get_model(weights)
    das Modell holen und model.predict(...) auf RGB-Bildern aufrufen (conf,
    verbose=False, device="cpu"). Aus jedem Ergebnis die Personen lesen
    (mit eurem `_persons_of`, siehe unten) und mit person_to_row in Zeilen-dicts
    umwandeln. Rückgabe: {"pred": [...]} mit einer Liste von Personen-dicts
    pro Bild, in derselben Reihenfolge wie der Batch.
    """
    raise NotImplementedError


def confident(p: Person, kp_conf: float = KP_CONF) -> Person:
    """Rein. Gibt eine neue Person zurück, bei der unsichere Punkte nicht zählen.

    Punkte mit Konfidenz (v) kleiner als kp_conf bekommen v = 0.0 (x und y
    bleiben), sichere Punkte bleiben unverändert. Box und score bleiben gleich.
    Hintergrund: Bei Vorhersagen ist v die Konfidenz und praktisch nie genau 0,
    ohne diesen Schritt gilt jeder geratene Punkt als markiert.
    """
    raise NotImplementedError


def _persons_of(result) -> tuple[Person, ...]:
    """Dünne Schale: ein YOLO-Ergebnis in Personen.

    Liest die vier Felder aus dem Ergebnis (siehe Anleitung Kapitel 1):
    result.boxes.xywhn, result.boxes.conf, result.keypoints.xyn,
    result.keypoints.conf. Es sind Tensoren: mit .tolist() in Listen umwandeln.
    result.keypoints.conf kann None sein. Dann result_to_persons aufrufen.
    """
    raise NotImplementedError


# --- vorgegeben ---------------------------------------------------------------


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
    """Unrein: ein einzelnes Bild (PIL; numpy-Arrays gehen über Image.fromarray)."""
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
    """Die Beobachtungen aus 7.2: map-Cache, conf ändern, Modellobjekt als Argument.

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
