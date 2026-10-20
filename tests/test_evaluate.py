from hypothesis import given, strategies as st

from pipeline.data import person_to_row
from pipeline.evaluate import (
    ZERO, Counts, add_counts, image_counts, match, pck, person_counts,
)
from pipeline.filters import labelled
from pipeline.types import Keypoint, Person

counts = st.builds(Counts, st.integers(0, 10_000), st.integers(0, 10_000))


def person(points: dict[int, tuple[float, float, float]], box=(50, 50, 100, 80)) -> Person:
    """Person mit 17 Keypoints; nicht genannte Indizes sind (0, 0, 0)."""
    kps = tuple(Keypoint(*points.get(i, (0.0, 0.0, 0.0))) for i in range(17))
    return Person(box, kps)


GT = person({0: (10, 10, 2), 1: (50, 50, 2)})  # 2 markierte Punkte, Box 100 x 80 -> Grösse 100


def test_zero_is_neutral_left_and_right():
    c = Counts(3, 7)
    assert add_counts(ZERO, c) == c
    assert add_counts(c, ZERO) == c


@given(counts, counts, counts)
def test_add_counts_associative(a, b, c):
    assert add_counts(a, add_counts(b, c)) == add_counts(add_counts(a, b), c)


@given(counts, counts)
def test_add_counts_commutative(a, b):
    assert add_counts(a, b) == add_counts(b, a)


def test_person_counts_without_prediction():
    assert person_counts(GT, None, 0.1) == Counts(0, labelled(GT))


def test_pck_of_zero_is_none():
    assert pck(ZERO) is None
    assert pck(Counts(1, 4)) == 0.25


def test_hand_example_one_near_one_far():
    pred = person({0: (12, 10, 0.9), 1: (500, 500, 0.9)})  # 2 px nahe, Rest weit weg
    assert person_counts(GT, pred, 0.1) == Counts(1, 2)


def test_only_marked_points_count_and_confidence_is_ignored():
    pred = person({0: (10, 10, 0.0), 1: (50, 50, 0.0), 5: (999, 999, 1.0)})
    assert person_counts(GT, pred, 0.1) == Counts(2, 2)


def test_threshold_is_alpha_times_larger_box_side():
    pred = person({0: (19, 10, 1.0), 1: (50, 50, 1.0)})  # Abstand 9 <= 0.1 * 100
    assert person_counts(GT, pred, 0.1) == Counts(2, 2)
    pred = person({0: (21, 10, 1.0), 1: (50, 50, 1.0)})  # Abstand 11 > 10
    assert person_counts(GT, pred, 0.1) == Counts(1, 2)


def test_match_picks_closest_and_handles_empty():
    near = person({0: (11, 10, 1), 1: (50, 51, 1)})
    far = person({0: (90, 90, 1), 1: (90, 90, 1)})
    assert match(GT, (far, near)) is near
    assert match(GT, ()) is None
    assert match(person({}), (near,)) is None  # Label ohne markierte Punkte


def test_image_counts_converts_normalised_to_pixels():
    # Bild 200 x 100. Normalisierte Werte; Label-Box 0.5 x 0.8 -> 100 x 80 px, Grösse 100.
    gt = person({0: (0.05, 0.1, 2), 1: (0.25, 0.5, 2)}, box=(0.25, 0.5, 0.5, 0.8))
    pred = person({0: (0.06, 0.1, 1), 1: (0.9, 0.9, 1)})  # Punkt 0: 2 px, Punkt 1: weit
    row = {
        "width": 200, "height": 100,
        "persons": [person_to_row(gt)], "pred": [person_to_row(pred)],
    }
    assert image_counts(row, 0.1) == Counts(1, 2)
    assert image_counts({**row, "pred": []}, 0.1) == Counts(0, 2)
    assert image_counts({**row, "persons": []}, 0.1) == ZERO
