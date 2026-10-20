from hypothesis import given, strategies as st

from pipeline.filters import all_of, angle_or, knee_angles, min_labelled
from pipeline.types import Keypoint, Person


def person_with(n_labelled: int) -> Person:
    kps = tuple(Keypoint(1.0, 1.0, 2.0 if i < n_labelled else 0.0) for i in range(17))
    return Person(box=(0.0, 0.0, 1.0, 1.0), keypoints=kps)


def test_all_of_empty_is_true():
    assert all_of()(42) is True


@given(st.integers(), st.integers(), st.integers())
def test_all_of_equals_and(a, b, x):
    p = lambda v: v > a
    q = lambda v: v < b
    assert all_of(p, q)(x) == (p(x) and q(x))


def test_min_labelled_zero_is_true_for_every_person():
    assert all(min_labelled(0)(person_with(n)) is True for n in range(18))


def test_min_labelled_threshold():
    assert min_labelled(8)(person_with(8)) is True
    assert min_labelled(8)(person_with(7)) is False


def test_knee_angles_without_labelled_legs_is_empty():
    assert knee_angles((person_with(0), person_with(0))) == ()


def test_angle_or_default_on_none():
    assert angle_or(None, 180.0) == 180.0
    assert angle_or(90.0, 180.0) == 90.0
    assert angle_or(0.0, 180.0) == 0.0  # 0.0 ist ein gültiger Wert, kein "fehlt"
