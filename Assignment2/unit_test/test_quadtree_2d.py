import numpy as np
import pytest
from spatial_trees.quadtree_2d import QuadTree2D

def brute_force_nn(points, target):
    d2 = np.sum((points - target) ** 2, axis=1)
    i = np.argmin(d2)
    return int(i), float(d2[i])

def test_nearest_neighbor_correctness():
    rng = np.random.default_rng(0)
    points = rng.random((200, 2))
    qt = QuadTree2D(points)

    for _ in range(30):
        q = rng.random(2)
        bi, bd2 = brute_force_nn(points, q)
        ti, td2 = qt.nearest((float(q[0]), float(q[1])))
        assert ti == bi
        assert abs(td2 - bd2) < 1e-12

def test_range_query_correctness():
    rng = np.random.default_rng(1)
    pts = rng.random((250, 2))
    qt = QuadTree2D(pts)

    rect = (0.1, 0.2, 0.6, 0.7)
    result = qt.range_query(rect)

    brute = []
    for i, (x, y) in enumerate(pts):
        if 0.1 <= x <= 0.6 and 0.2 <= y <= 0.7:
            brute.append(i)

    assert sorted(result) == sorted(brute)

def test_small_medium_scaling():
    rng = np.random.default_rng(2)
    for n in [100, 1000, 5000]:
        pts = rng.random((n, 2))
        qt = QuadTree2D(pts)
        idx, d2 = qt.nearest((0.4, 0.4))
        assert 0 <= idx < n
        assert d2 >= 0.0

def test_no_points_behavior():
    qt = QuadTree2D(np.zeros((0, 2)))
    idx, d2 = qt.nearest((0.1, 0.2))
    assert idx == -1
    assert d2 == float("inf")
    assert qt.range_query((0, 0, 1, 1)) == []
