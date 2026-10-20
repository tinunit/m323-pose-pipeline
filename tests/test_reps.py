from hypothesis import given, strategies as st

from pipeline.reps import START, RepState, count_reps, drop_missing, ema, states, step

NOISY = [170, 132, 128, 131, 127, 90, 129, 133, 128, 135, 170]
SQUAT = [170] * 5 + [80] * 5 + [170] * 5


def test_leer():
    assert count_reps([]) == 0


def test_zwei_kniebeugen():
    assert count_reps([170, 90, 165, 85, 170]) == 2


def test_nicht_ganz_aufgestanden():
    assert count_reps([170, 95, 150, 95, 170]) == 1


def test_noisy_zwei_schwellen():
    assert count_reps(NOISY) == 1


def test_ema_squat():
    assert count_reps(ema(SQUAT, 0.5)) == 1
    assert count_reps(ema(SQUAT, 0.2)) == 0


def test_deterministisch():
    xs = [170, 90, 165, 85, 170]
    assert count_reps(xs) == count_reps(xs) == 2


def test_ema_leer():
    assert ema([], 0.5) == []


def test_step_und_states():
    assert step(START, 90) == RepState("down", 0)
    assert step(RepState("down", 0), 170) == RepState("up", 1)
    assert step(START, 130) == START
    assert states([]) == [START]
    assert states([90, 170])[-1] == RepState("up", 1)


def test_drop_missing():
    assert drop_missing([1.0, None, 2.0]) == [1.0, 2.0]


angles = st.lists(st.floats(0, 180), max_size=60)


@given(angles)
def test_property_count_hoechstens_runterkreuzungen(xs):
    downs = sum(1 for a in xs if a < 100)
    assert 0 <= count_reps(xs) <= downs


@given(angles)
def test_property_deterministisch_und_pure(xs):
    before = list(xs)
    assert count_reps(xs) == count_reps(xs)
    assert xs == before
    assert states(xs)[-1].count == count_reps(xs)


@given(angles, st.floats(0.05, 1.0))
def test_property_ema_laenge_und_bereich(xs, alpha):
    s = ema(xs, alpha)
    assert len(s) == len(xs)
    if xs:
        assert all(min(xs) - 1e-9 <= v <= max(xs) + 1e-9 for v in s)
