import math

from hypothesis import given, strategies as st
from pytest import approx

from pipeline.draw import draw_skeleton
from pipeline.geometry import angle_at, knee_angle, to_pixels
from pipeline.types import Keypoint as K, Person
from PIL import Image


def person(**kps: K) -> Person:
    """Person mit 17 leeren Keypoints, einzelne per Index (k13=...) gesetzt."""
    ks = [K(0, 0, 0)] * 17
    for name, k in kps.items():
        ks[int(name[1:])] = k
    return Person(box=(0.5, 0.5, 0.2, 0.4), keypoints=tuple(ks))


def test_angle_straight():
    assert angle_at(K(0, 0, 2), K(0, 1, 2), K(0, 2, 2)) == approx(180)


def test_angle_right():
    assert angle_at(K(0, 0, 2), K(0, 1, 2), K(1, 1, 2)) == approx(90)


def test_angle_degenerate_is_none():
    p, q = K(1, 1, 2), K(2, 3, 2)
    assert angle_at(p, p, q) is None
    assert angle_at(q, p, p) is None


def test_angle_rounding_does_not_raise():
    # fast kollinear: cos kann leicht ausserhalb [-1, 1] landen
    a, b, c = K(0.1, 0.1, 2), K(0.3, 0.3, 2), K(0.7, 0.7, 2)
    assert angle_at(a, b, c) == approx(180)


def test_to_pixels_values_and_input_unchanged():
    p = person(k11=K(0.5, 0.5, 2))
    before = p
    q = to_pixels(p, 200, 100)
    assert p == before and p.keypoints[11] == K(0.5, 0.5, 2)
    assert q.keypoints[11] == K(100, 50, 2)
    assert q.box == (100, 50, 40, 40)
    assert q.keypoints[0] == K(0, 0, 0) and q.score == p.score


def test_knee_angle_none_if_unlabelled():
    p = person(k11=K(0, 0, 2), k13=K(0, 1, 2), k15=K(0, 2, 0))
    assert knee_angle(p, "left") is None


def test_knee_angle_sides():
    p = person(k11=K(0, 0, 2), k13=K(0, 1, 2), k15=K(0, 2, 2),
               k12=K(0, 0, 2), k14=K(0, 1, 2), k16=K(1, 1, 2))
    assert knee_angle(p, "left") == approx(180)
    assert knee_angle(p, "right") == approx(90)


def test_draw_skeleton_returns_new_image_and_keeps_input():
    img = Image.new("RGB", (100, 100), "white")
    before = img.tobytes()
    p = person(k11=K(10, 10, 2), k13=K(50, 50, 2), k15=K(80, 90, 2))
    out = draw_skeleton(img, p)
    assert out is not img
    assert img.tobytes() == before
    assert out.tobytes() != before


def test_draw_skeleton_skips_unlabelled():
    img = Image.new("RGB", (100, 100), "white")
    p = person(k11=K(10, 10, 2), k13=K(50, 50, 0))
    out = draw_skeleton(img, p)
    assert out.getpixel((50, 50)) == (255, 255, 255)  # Punkt nicht gezeichnet
    assert out.getpixel((30, 30)) == (255, 255, 255)  # Kante nicht gezeichnet
    assert out.getpixel((10, 10)) != (255, 255, 255)  # markierter Punkt schon


# Koordinaten in [-100, 100] in Schritten, die Punkte b weit genug von a und c
# weg halten (Länge >= 1). Toleranz 1e-3 Grad: nahe 0 Grad verstärkt acos
# Rundungsfehler (1e-16 im cos wird zu ~1e-6 Grad).
coord = st.floats(-100, 100, allow_nan=False, allow_infinity=False)
point = st.builds(lambda x, y: K(x, y, 2), coord, coord)


def far(a: K, b: K) -> bool:
    return math.hypot(a.x - b.x, a.y - b.y) >= 1


@given(point, point, point, st.floats(0.01, 100))
def test_angle_scale_invariant(a, b, c, k):
    from hypothesis import assume
    assume(far(a, b) and far(c, b))
    scale = lambda q: K(q.x * k, q.y * k, q.v)
    assert angle_at(scale(a), scale(b), scale(c)) == approx(angle_at(a, b, c), abs=1e-3)  # acos ist nahe 0 Grad empfindlich
