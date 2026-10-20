"""Tests für Schritt 4 (ohne Modell: nur der reine Kern)."""

from hypothesis import given, strategies as st

from pipeline.inference import FEATURES_WITH_PRED, result_to_persons
from pipeline.types import Keypoint, Person

XY = [[0.1 * i, 0.05 * i] for i in range(17)]
CONF = [0.5 + 0.01 * i for i in range(17)]


def test_keine_erkennung():
    assert result_to_persons([], [], [], None) == ()


def test_eine_person_17_keypoints():
    ps = result_to_persons([[0.5, 0.5, 0.2, 0.8]], [0.9], [XY], [CONF])
    assert len(ps) == 1
    p = ps[0]
    assert isinstance(p, Person)
    assert len(p.keypoints) == 17
    assert p.box == (0.5, 0.5, 0.2, 0.8)
    assert p.score == 0.9
    assert p.keypoints[3] == Keypoint(XY[3][0], XY[3][1], CONF[3])


def test_kps_conf_none_gibt_v_eins():
    ps = result_to_persons([[0.5, 0.5, 0.2, 0.8]], [0.9], [XY], None)
    assert all(k.v == 1.0 for k in ps[0].keypoints)


def test_mehrere_personen_reihenfolge():
    ps = result_to_persons(
        [[0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.1, 0.1]], [0.3, 0.8], [XY, XY], [CONF, CONF]
    )
    assert [p.score for p in ps] == [0.3, 0.8]


def test_features_with_pred_hat_gleichen_aufbau_wie_persons():
    assert FEATURES_WITH_PRED["pred"] == FEATURES_WITH_PRED["persons"]
    assert "image" in FEATURES_WITH_PRED


@given(st.integers(min_value=0, max_value=5), st.booleans())
def test_anzahl_personen_bleibt_erhalten(n, with_conf):
    boxes = [[0.5, 0.5, 0.1, 0.1]] * n
    ps = result_to_persons(boxes, [0.5] * n, [XY] * n, [CONF] * n if with_conf else None)
    assert len(ps) == n
    assert all(len(p.keypoints) == 17 for p in ps)
