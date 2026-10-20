import pytest
from hypothesis import given, strategies as st
from PIL import Image

from datasets import Dataset

from pipeline.augment import (
    FLIP_INDEX,
    hflip_person,
    hflip_row,
    make_transform,
    seed_for,
)
from pipeline.data import labelled_count, person_to_row
from pipeline.types import Keypoint, Person

unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
keypoint = st.builds(Keypoint, unit, unit, st.sampled_from([0.0, 1.0, 2.0]))
person = st.builds(
    Person,
    box=st.tuples(unit, unit, unit, unit),
    keypoints=st.tuples(*[keypoint] * 17),
    score=unit,
)


def make_person() -> Person:
    kps = tuple(Keypoint(0.1 + 0.05 * i, 0.5, 2.0 if i % 3 else 0.0) for i in range(17))
    return Person(box=(0.4, 0.5, 0.2, 0.6), keypoints=kps)


def test_struktur_links_rechts_getauscht():
    p = make_person()
    f = hflip_person(p)
    for i, k in enumerate(p.keypoints):
        if k.v > 0:
            assert f.keypoints[FLIP_INDEX[i]].x == pytest.approx(1 - k.x)
            assert f.keypoints[FLIP_INDEX[i]].y == k.y


def test_anzahl_markierter_keypoints_bleibt():
    p = make_person()
    assert labelled_count(hflip_person(p)) == labelled_count(p)


def test_box_cx_wird_gespiegelt():
    assert hflip_person(make_person()).box == pytest.approx((0.6, 0.5, 0.2, 0.6))


@given(person)
def test_zweimal_spiegeln_ergibt_original(p):
    back = hflip_person(hflip_person(p))
    assert back.score == p.score
    assert back.box == pytest.approx(p.box)
    for a, b in zip(back.keypoints, p.keypoints):
        assert a == pytest.approx(b)


def test_seed_for_formel():
    assert seed_for(42, 0, 3) == 42 * 1_000_003 + 3
    assert seed_for(42, 1, 3) == 42 * 1_000_003 + 1_009 + 3


def _row(w=4, h=2):
    img = Image.new("RGB", (w, h))
    img.putpixel((0, 0), (255, 0, 0))
    return {
        "image": img,
        "persons": [person_to_row(make_person())],
        "pred": [1],
        "idx": 0,
    }


def test_hflip_row_spiegelt_bild_und_entfernt_pred():
    out = hflip_row(_row())
    assert "pred" not in out
    assert out["image"].getpixel((3, 0)) == (255, 0, 0)
    assert out["persons"][0]["box"][0] == pytest.approx(0.6)


def test_with_transform_ist_deterministisch_und_veraendert_ds_nicht():
    ds = Dataset.from_list(
        [{"image": _row()["image"], "persons": [person_to_row(make_person())], "idx": i}
         for i in range(8)]
    )
    ds_t = ds.with_transform(make_transform(42, 0))
    assert ds_t[0] == ds_t[0]
    assert ds_t[:8] == ds_t[:8]
    assert ds[0]["persons"][0]["box"][0] == pytest.approx(0.4)  # Original unverändert
    assert hasattr(ds, "with_transform") and ds_t is not ds
