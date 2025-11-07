import numpy as np
import pytest
from spatial_trees.implicit_kd_tree_2d import ImplicitKDTree2D

def brute_force_nn(points, target):
    d2 = np.sum((points - target) ** 2, axis=1)
    i = np.argmin(d2)
    return int(i), float(d2[i])

@pytest.mark.parametrize("split_method", ["variance", "interleave"])
@pytest.mark.parametrize("approx_median", [False, True])
def test_nearest_neighbor_correctness(split_method, approx_median):
    rng = np.random.default_rng(0)
    points = rng.random((200, 2))

    tree = ImplicitKDTree2D(
        points,
        split_method=split_method,
        approx_median=approx_median,
        sample_size=64
    )

    for _ in range(20):
        q = rng.random(2)
        bi, bd2 = brute_force_nn(points, q)
        ti, td2 = tree.nearest((float(q[0]), float(q[1])))
        assert ti == bi
        assert abs(td2 - bd2) < 1e-12

def test_range_query_correctness():
    rng = np.random.default_rng(1)
    pts = rng.random((300, 2))
    tree = ImplicitKDTree2D(pts)

    rect = (0.2, 0.2, 0.7, 0.8)
    qi = tree.range_query(rect)

    # brute force
    bf = []
    for i, (x, y) in enumerate(pts):
        if 0.2 <= x <= 0.7 and 0.2 <= y <= 0.8:
            bf.append(i)

    assert sorted(qi) == sorted(bf)

def test_small_medium_scaling():
    rng = np.random.default_rng(2)
    for n in [100, 1000, 5000]:
        pts = rng.random((n, 2))
        tree = ImplicitKDTree2D(pts)
        idx, d2 = tree.nearest((0.3, 0.6))
        assert 0 <= idx < n
        assert d2 >= 0.0

def test_structure_consistency_between_modes():
    rng = np.random.default_rng(3)
    pts = rng.random((50, 2))
    t1 = ImplicitKDTree2D(pts, split_method="variance")
    t2 = ImplicitKDTree2D(pts, split_method="interleave")
    assert len(t1.perm) == len(t2.perm)
