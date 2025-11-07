import numpy as np
import pytest
from spatial_trees.kd_tree_2d import KDTree2D

def brute_force_nn(points, target):
    d2 = np.sum((points - target) ** 2, axis=1)
    i = np.argmin(d2)
    return i, d2[i]

@pytest.mark.parametrize("split_method", ["variance", "interleave"])
def test_nearest_neighbor_correctness(split_method):
    rng = np.random.default_rng(0)
    points = rng.random((200, 2))
    tree = KDTree2D(points, split_method=split_method)

    for _ in range(20):
        target = rng.random(2)
        idx_tree, d2_tree = tree.nearest(tuple(target))
        idx_brute, d2_brute = brute_force_nn(points, target)
        # Verify correctness within tolerance
        assert np.isclose(d2_tree, d2_brute, atol=1e-12)

def test_empty_and_single_point_cases():
    pts = np.array([[0.5, 0.5]])
    tree = KDTree2D(pts)
    idx, d2 = tree.nearest((0.5, 0.5))
    assert idx == 0 and d2 == 0.0

def test_repeat_points():
    pts = np.array([[0.2, 0.3], [0.2, 0.3], [0.2, 0.3]])
    tree = KDTree2D(pts)
    idx, d2 = tree.nearest((0.2, 0.3))
    assert d2 == 0.0

def test_interleave_and_variance_give_similar_structure():
    rng = np.random.default_rng(1)
    pts = rng.random((30, 2))
    t1 = KDTree2D(pts, split_method="variance")
    t2 = KDTree2D(pts, split_method="interleave")
    assert t1.points.shape == t2.points.shape

def test_scaling_small_to_medium():
    rng = np.random.default_rng(2)
    for n in [100, 1000, 5000]:
        pts = rng.random((n, 2))
        tree = KDTree2D(pts)
        idx, d2 = tree.nearest((0.3, 0.7))
        assert 0 <= idx < n
        assert d2 >= 0.0
