from PIL import Image

from pipeline.geometry import knee_angle
from pipeline.inference import confident
from pipeline.types import Keypoint, Person
from run_own import normalize_image


def _person(knee_conf: float) -> Person:
    kps = [Keypoint(0, 0, 0.9)] * 17
    kps[11], kps[13], kps[15] = Keypoint(0, 0, 0.9), Keypoint(0, 1, knee_conf), Keypoint(1, 1, 0.9)
    return Person((0.5, 0.5, 1, 1), tuple(kps), 0.9)


def test_confident_hides_unsure_knee():
    assert knee_angle(_person(0.05), "left") is not None  # ohne Schwelle: geratener Winkel
    assert knee_angle(confident(_person(0.05)), "left") is None
    assert knee_angle(confident(_person(0.8)), "left") is not None


def test_confident_keeps_input():
    p = _person(0.05)
    confident(p)
    assert p.keypoints[13].v == 0.05


def test_normalize_applies_exif_rotation():
    im = Image.new("RGB", (400, 200))  # gespeichert quer
    ex = Image.Exif()
    ex[0x0112] = 6  # anzeigen: 90 Grad drehen
    im.info["exif"] = ex.tobytes()
    im.getexif()[0x0112] = 6
    assert normalize_image(im).size == (200, 400)  # aufrecht


def test_normalize_scales_and_converts():
    out = normalize_image(Image.new("RGBA", (4000, 3000)), max_side=1280)
    assert out.size == (1280, 960)
    assert out.mode == "RGB"
    assert normalize_image(Image.new("L", (100, 50))).size == (100, 50)  # kleine bleiben
