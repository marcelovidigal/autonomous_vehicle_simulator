import numpy as np
import pytest

from core import Config, Track

NAMES = Track.list_names()


def test_tracks_exist():
    assert "circuito_1" in NAMES


@pytest.mark.parametrize("name", NAMES)
def test_load_and_start_on_track(name):
    t = Track.load(name)
    s = t.start_state()
    _, dist = t.locate(s.pos)
    assert dist <= t.crash_dist
    assert t.length > 100.0
    assert t.outer.shape == t.inner.shape


@pytest.mark.parametrize("name", NAMES)
def test_sensor_readings_shape_and_range(name):
    t = Track.load(name)
    s = t.start_state()
    r = t.sensor_readings(s.pos, s.heading, Config().sensors)
    assert r.shape == (Config().sensors.count,)
    assert np.all(r >= 0.0) and np.all(r <= 1.0)


def test_offset_curvature_ok_no_gross_selfintersection():
    # área do polígono externo deve ser maior que a do interno e ambas positivas
    for name in NAMES:
        t = Track.load(name)
        from core.geometry import polygon_area

        assert polygon_area(t.outer) > polygon_area(t.inner) > 0.0


def test_commit_progress_advances_and_laps_wrap():
    t = Track.load("circuito_1")
    s = t.start_state()
    n = len(t.center)
    # avança alguns índices
    t.commit_progress(s, (s.idx + 5) % n)
    assert s.progress > 0.0
    # simula a virada de volta (índice quase no fim -> início)
    s.idx = n - 3
    t.commit_progress(s, 2)
    assert s.laps == 1 and s.finished
