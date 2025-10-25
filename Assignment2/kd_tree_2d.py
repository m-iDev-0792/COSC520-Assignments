# KD-Tree (2D) implementation with two splitting strategies (variance-based, interleaving)
# plus a small benchmarking harness for time & memory.
#
# You can tweak the "if __name__ == '__main__':" section at the bottom to try bigger sizes.
#
# Notes:
# - Queries are 1-NN (nearest neighbor) in Euclidean distance.
# - "0.2 * N" queries are performed on random targets sampled from the data distribution.
# - Build memory is measured via tracemalloc peak during build.
# - Build/Query time is measured with time.perf_counter().
#
# The script also runs a quick demo benchmark so you can see results immediately.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal
import math
import random
import time
import tracemalloc
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
from dataset_load import *
from typing import Union, List, Any

# Define the acceptable input types for clarity
DataTypes = Union[List[Any], np.ndarray]

def sample_n_elements(data: DataTypes, n: int, allow_duplicates: bool = False) -> DataTypes:
    """
    Randomly samples N elements (or rows, for 2D arrays) from a list or a NumPy array 
    """
    data_length = len(data)

    if n > data_length and not allow_duplicates:
        raise ValueError(f"Cannot sample {n} unique items from a structure of size {data_length}. Set allow_duplicates=True.")

    if isinstance(data, list):
        if allow_duplicates:
            return random.choices(data, k=n)
        else:
            return random.sample(data, k=n)

    elif isinstance(data, np.ndarray):        
        num_items_to_sample = data.shape[0]
        indices = np.random.choice(
            num_items_to_sample, 
            size=n, 
            replace=allow_duplicates
        )
        return data[indices]

    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Must be a list or a numpy.ndarray.")



SplitMethod = Literal["variance", "interleave"]


@dataclass
class KDNode2D:
    point_idx: int
    axis: int
    left: Optional["KDNode2D"]
    right: Optional["KDNode2D"]
    bounds: Tuple[float, float, float, float]


class KDTree2D:
    """
    2D KD-Tree with variance- or interleaving-based splits.
    Supports optional bounding-box pruning for faster nearest-neighbor queries.
    """

    def __init__(self, points: np.ndarray, split_method: SplitMethod = "variance", seed: Optional[int] = None):
        assert points.ndim == 2 and points.shape[1] == 2, "points must be Nx2"
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.split_method = split_method
        idxs = np.arange(points.shape[0], dtype=np.int64)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        self.root = self._build(idxs, 0, (x_min, x_max, y_min, y_max))

    def _choose_axis(self, pts_idxs: np.ndarray, depth: int) -> int:
        if self.split_method == "interleave":
            return depth & 1
        subset = self.points[pts_idxs]
        return 0 if np.var(subset[:, 0]) >= np.var(subset[:, 1]) else 1

    def _build(self, pts_idxs: np.ndarray, depth: int, bounds) -> Optional[KDNode2D]:
        if pts_idxs.size == 0:
            return None

        axis = self._choose_axis(pts_idxs, depth)
        median_pos = pts_idxs.size // 2
        local = pts_idxs.copy()
        coords = self.points[local, axis]
        median_idx_in_local = np.argpartition(coords, median_pos)[median_pos]
        median_point_idx = local[median_idx_in_local]

        px, py = self.points[median_point_idx]
        x0, x1, y0, y1 = bounds
        if axis == 0:
            left_bounds = (x0, px, y0, y1)
            right_bounds = (px, x1, y0, y1)
        else:
            left_bounds = (x0, x1, y0, py)
            right_bounds = (x0, x1, py, y1)

        left_mask = coords < coords[median_idx_in_local]
        right_mask = coords > coords[median_idx_in_local]
        eq_mask = ~(left_mask | right_mask)
        eq_mask[np.where(np.arange(local.size) == median_idx_in_local, True, False)] = False
        eq_idxs = local[eq_mask]
        half = eq_idxs.size // 2

        left_idxs = np.concatenate([local[left_mask], eq_idxs[:half]])
        right_idxs = np.concatenate([local[right_mask], eq_idxs[half:]])

        return KDNode2D(
            point_idx=int(median_point_idx),
            axis=int(axis),
            left=self._build(left_idxs, depth + 1, left_bounds),
            right=self._build(right_idxs, depth + 1, right_bounds),
            bounds=bounds,
        )

    # ---------------- QUERY ----------------
    def nearest(self, target: Tuple[float, float], use_prune: bool = True) -> Tuple[int, float]:
        best_idx = -1
        best_d2 = float("inf")

        def dist2_point(i: int):
            dx = self.points[i, 0] - target[0]
            dy = self.points[i, 1] - target[1]
            return dx * dx + dy * dy

        def min_dist2_to_box(bounds):
            x0, x1, y0, y1 = bounds
            px, py = target
            dx = 0.0 if x0 <= px <= x1 else min(abs(px - x0), abs(px - x1))
            dy = 0.0 if y0 <= py <= y1 else min(abs(py - y0), abs(py - y1))
            return dx * dx + dy * dy

        def search(node: Optional[KDNode2D]):
            nonlocal best_idx, best_d2
            if node is None:
                return
            if use_prune and min_dist2_to_box(node.bounds) >= best_d2:
                return
            d2 = dist2_point(node.point_idx)
            if d2 < best_d2:
                best_idx = node.point_idx
                best_d2 = d2
            axis = node.axis
            pivot = self.points[node.point_idx, axis]
            val = target[axis]
            near, far = (node.left, node.right) if val < pivot else (node.right, node.left)
            search(near)
            if not use_prune or (val - pivot) ** 2 < best_d2:
                search(far)

        search(self.root)
        return best_idx, best_d2


# ---------------- BENCHMARKING ----------------
def benchmark_kdtree():
    N_values = [1000, 5000, 10000, 20000, 50000]
    results = []

    # load dataset
    print(f'loading points from directory ./dataset')
    points = load_points("./dataset")
    points_num = len(points)
    print(f'loaded {points_num} points')
    query_num_ration = 0.2
    source_point_num = points_num - round(points_num * query_num_ration)
    source_point_set = points[:source_point_num]
    query_point_set = points[source_point_num:]
    print(f'size of source points = {len(source_point_set)}, size of query points = {len(query_point_set)}')
    for N in N_values:
        points = sample_n_elements(source_point_set, N)
        queries = sample_n_elements(query_point_set, round(query_num_ration * N))
        for method in ["variance", "interleave"]:
            for use_prune in [False, True]:
                # measure build
                tracemalloc.start()
                t0 = time.perf_counter()
                tree = KDTree2D(points, split_method=method)
                build_time = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                # measure query
                t1 = time.perf_counter()
                for q in queries:
                    tree.nearest(tuple(q), use_prune=use_prune)
                query_time = time.perf_counter() - t1

                results.append({
                    "N": N,
                    "method": method,
                    "prune": use_prune,
                    "build_time_s": build_time,
                    "query_time_s": query_time,
                    "peak_MB": peak / 1e6,
                })
                print(f"✓ N={N:<6} {method:<11} prune={use_prune:<5} build={build_time:.4f}s query={query_time:.4f}s")

    df = pd.DataFrame(results)
    df.to_csv("./output/kdtree2d_prune_benchmark.csv", index=False)
    print("\n✅ Results saved to ./output/kdtree2d_prune_benchmark.csv")

    # -------- Plot results --------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(
        axes,
        ["build_time_s", "query_time_s", "peak_MB"],
        ["Build Time (s)", "Query Time (s)", "Peak Memory (MB)"]
    ):
        for method in ["variance", "interleave"]:
            subset = df[df["method"] == method]
            ax.plot(subset[subset["prune"] == False]["N"], subset[subset["prune"] == False][metric],
                    "--o", label=f"{method}-no prune")
            ax.plot(subset[subset["prune"] == True]["N"], subset[subset["prune"] == True][metric],
                    "-o", label=f"{method}-prune")
        ax.set_xlabel("N (points)")
        ax.set_ylabel(title)
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plt.savefig("./output/kdtree2d_prune_benchmark.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    benchmark_kdtree()
