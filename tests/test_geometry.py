import numpy as np

from core import geometry as geo

SQUARE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])


def test_point_in_polygon():
    assert geo.point_in_polygon((1.0, 1.0), SQUARE)
    assert not geo.point_in_polygon((3.0, 1.0), SQUARE)
    assert not geo.point_in_polygon((-0.1, 1.0), SQUARE)


def test_polyline_length_square():
    assert abs(geo.polyline_length(SQUARE, closed=True) - 8.0) < 1e-9


def test_resample_closed_preserves_length_and_count():
    pts = geo.resample_closed(SQUARE, 100)
    assert pts.shape == (100, 2)
    assert abs(geo.polyline_length(pts, closed=True) - 8.0) < 1e-6


def test_offset_areas_differ_by_sign():
    a = geo.polygon_area(geo.offset_closed(SQUARE, 0.5))
    b = geo.polygon_area(geo.offset_closed(SQUARE, -0.5))
    assert max(a, b) > 5.0 > min(a, b) > 0.0


def test_ray_fan_hits_known_segment():
    # raio de (0,0) para +x contra o segmento vertical x=10, y in [-5,5]
    seg_a = np.array([[10.0, -5.0]])
    seg_b = np.array([[10.0, 5.0]])
    d = geo.ray_fan_distances((0.0, 0.0), [[1.0, 0.0]], seg_a, seg_b, max_dist=100.0)
    assert abs(d[0] - 10.0) < 1e-9


def test_ray_fan_miss_returns_max():
    seg_a = np.array([[10.0, -5.0]])
    seg_b = np.array([[10.0, 5.0]])
    d = geo.ray_fan_distances((0.0, 0.0), [[-1.0, 0.0]], seg_a, seg_b, max_dist=42.0)
    assert d[0] == 42.0


def test_fan_angles_symmetric():
    a = geo.fan_angles(5, 160.0)
    assert a.shape == (5,)
    assert abs(a[0] + a[-1]) < 1e-12
    assert abs(a[2]) < 1e-12
    assert geo.fan_angles(1, 160.0).tolist() == [0.0]
