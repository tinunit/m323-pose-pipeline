from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from pipeline.data import (
    labelled_count,
    load_split,
    parse_label_file,
    parse_label_line,
    person_to_row,
    row_to_person,
)
from pipeline.types import Keypoint, Person

ROOT = Path(__file__).resolve().parent.parent / "data" / "coco8-pose"

coord = st.floats(0, 1, width=32)
keypoints = st.tuples(*[
    st.builds(Keypoint, coord, coord, st.sampled_from([0.0, 1.0, 2.0]))
    for _ in range(17)
])
persons = st.builds(
    Person,
    box=st.tuples(coord, coord, coord, coord),
    keypoints=keypoints,
    score=coord,
)


def test_zero_line_gives_17_unlabelled_keypoints():
    p = parse_label_line("0 0.5 0.5 0.2 0.2 " + "0 0 0 " * 17)
    assert len(p.keypoints) == 17
    assert all(k.v == 0 for k in p.keypoints)


def test_55_numbers_gives_none():
    assert parse_label_line(" ".join(["0"] * 55)) is None


def test_empty_file_gives_empty_tuple():
    assert parse_label_file("") == ()


def test_invalid_lines_are_dropped():
    good = "0 0.5 0.5 0.2 0.2 " + "1 1 2 " * 17
    assert len(parse_label_file(good + "\nmüll\n" + good + "\n")) == 2


def test_roundtrip_example():
    p = parse_label_line("0 0.5 0.5 0.25 0.25 " + "0.5 0.25 2 " * 17)
    assert row_to_person(person_to_row(p)) == p


@given(persons)
def test_roundtrip_property(p):
    assert row_to_person(person_to_row(p)) == p


@given(persons)
def test_row_uses_lists(p):
    row = person_to_row(p)
    assert isinstance(row["box"], list)
    assert all(isinstance(k, list) for k in row["keypoints"])


def test_keypoint_is_immutable():
    p = Person(box=(0, 0, 1, 1), keypoints=tuple(Keypoint(0, 0, 0) for _ in range(17)))
    with pytest.raises(AttributeError):
        p.keypoints[0].x = 1


def test_labelled_count():
    p = parse_label_line("0 0.5 0.5 0.2 0.2 " + "1 1 2 " * 3 + "0 0 0 " * 14)
    assert labelled_count(p) == 3


@pytest.mark.parametrize("split", ["train", "val"])
def test_load_split_real_data_has_4_rows(split):
    assert len(load_split(ROOT, split)) == 4
