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
    Supports exact median or approximate median sampling for construction.
    """

    def __init__(self, points: np.ndarray, split_method: SplitMethod = "variance", 
                 approx_median: bool = False, sample_size: int = 100, seed: Optional[int] = None):
        """
        Args:
            points: Nx2 array of 2D points
            split_method: "variance" or "interleave" for axis selection
            approx_median: If True, use approximate median via sampling
            sample_size: Number of points to sample for approximate median (ignored if approx_median=False)
            seed: Random seed for reproducibility
        """
        assert points.ndim == 2 and points.shape[1] == 2, "points must be Nx2"
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.split_method = split_method
        self.approx_median = approx_median
        self.sample_size = sample_size
        
        if seed is not None:
            np.random.seed(seed)
        
        idxs = np.arange(points.shape[0], dtype=np.int64)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        self.root = self._build(idxs, 0, (x_min, x_max, y_min, y_max))

    def _choose_axis(self, pts_idxs: np.ndarray, depth: int) -> int:
        if self.split_method == "interleave":
            return depth & 1
        subset = self.points[pts_idxs]
        return 0 if np.var(subset[:, 0]) >= np.var(subset[:, 1]) else 1

    def _find_split_point(self, pts_idxs: np.ndarray, axis: int) -> int:
        """
        Find the median point index for splitting.
        Uses either exact median (argpartition) or approximate median (sampling).
        
        Returns:
            Index into pts_idxs array of the median point
        """
        n = pts_idxs.size
        
        if not self.approx_median or n <= self.sample_size:
            # Exact median using argpartition (O(n))
            coords = self.points[pts_idxs, axis]
            median_pos = n // 2
            median_idx_in_local = np.argpartition(coords, median_pos)[median_pos]
            return median_idx_in_local
        else:
            # Approximate median using sampling (O(k log k) where k = sample_size)
            # Sample random indices
            sample_indices = np.random.choice(n, size=self.sample_size, replace=False)
            sampled_pts = pts_idxs[sample_indices]
            sampled_coords = self.points[sampled_pts, axis]
            
            # Find median of sample
            sample_median_val = np.median(sampled_coords)
            
            # Find the point in the full set closest to this median value
            all_coords = self.points[pts_idxs, axis]
            closest_idx = np.argmin(np.abs(all_coords - sample_median_val))
            return closest_idx

    def _build(self, pts_idxs: np.ndarray, depth: int, bounds) -> Optional[KDNode2D]:
        if pts_idxs.size == 0:
            return None

        axis = self._choose_axis(pts_idxs, depth)
        local = pts_idxs.copy()
        coords = self.points[local, axis]
        
        # Find split point using exact or approximate median
        median_idx_in_local = self._find_split_point(local, axis)
        median_point_idx = local[median_idx_in_local]

        # Calculate bounds for subtrees
        px, py = self.points[median_point_idx]
        x0, x1, y0, y1 = bounds
        if axis == 0:
            left_bounds = (x0, px, y0, y1)
            right_bounds = (px, x1, y0, y1)
        else:
            left_bounds = (x0, x1, y0, py)
            right_bounds = (x0, x1, py, y1)

        # Partition points
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
    def nearest(self, target: Tuple[float, float]) -> Tuple[int, float]:
        """
        Find the nearest neighbor to target point.
        Uses standard KD-tree search with branch pruning.
        """
        best_idx = -1
        best_d2 = float("inf")

        tx, ty = target  # Unpack once
        
        def dist2_point(i: int):
            dx = self.points[i, 0] - tx
            dy = self.points[i, 1] - ty
            return dx * dx + dy * dy

        def min_dist2_to_box(bounds):
            x0, x1, y0, y1 = bounds
            dx = 0.0 if x0 <= tx <= x1 else min(abs(tx - x0), abs(tx - x1))
            dy = 0.0 if y0 <= ty <= y1 else min(abs(ty - y0), abs(ty - y1))
            return dx * dx + dy * dy

        def search(node: Optional[KDNode2D]):
            nonlocal best_idx, best_d2
            if node is None:
                return
            
            # Branch pruning
            if min_dist2_to_box(node.bounds) >= best_d2:
                return
            
            # Check current node
            d2 = dist2_point(node.point_idx)
            if d2 < best_d2:
                best_idx = node.point_idx
                best_d2 = d2
            
            # Determine search order
            axis = node.axis
            pivot = self.points[node.point_idx, axis]
            val = target[axis]
            
            if val < pivot:
                near, far = node.left, node.right
            else:
                near, far = node.right, node.left
            
            # Always search near side
            search(near)
            
            # Conditionally search far side
            if (val - pivot) ** 2 < best_d2:
                search(far)

        search(self.root)
        return best_idx, best_d2


class LinearSearch2D:
    """
    Simple linear (brute-force) search for nearest neighbor.
    Iterates through all points to find the closest one.
    """
    
    def __init__(self, points: np.ndarray):
        assert points.ndim == 2 and points.shape[1] == 2, "points must be Nx2"
        self.points = np.ascontiguousarray(points, dtype=np.float64)
    
    def nearest(self, target: Tuple[float, float]) -> Tuple[int, float]:
        """Find the nearest neighbor using brute-force search."""
        tx, ty = target
        dx = self.points[:, 0] - tx
        dy = self.points[:, 1] - ty
        distances_sq = dx * dx + dy * dy
        best_idx = np.argmin(distances_sq)
        best_d2 = distances_sq[best_idx]
        return int(best_idx), float(best_d2)


def benchmark_kdtree():
    N_values = [10000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 10000000]
    sample_size = 100  # Sample size for approximate median
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
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS: Exact vs Approximate Median")
    print("="*80)
    
    for N in N_values:
        points = sample_n_elements(source_point_set, N)
        queries = sample_n_elements(query_point_set, round(query_num_ration * N))
        
        print(f"\n--- N = {N} ({len(queries)} queries) ---")
        
        # Benchmark Linear Search
        tracemalloc.start()
        t0 = time.perf_counter()
        linear = LinearSearch2D(points)
        build_time = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        t1 = time.perf_counter()
        for q in queries:
            linear.nearest(tuple(q))
        query_time = time.perf_counter() - t1
        
        results.append({
            "N": N,
            "method": "linear",
            "approx_median": False,
            "sample_size": 0,
            "build_time_s": build_time,
            "query_time_s": query_time,
            "peak_MB": peak / 1e6,
        })
        print(f"  Linear:                    build={build_time:.4f}s  query={query_time:.4f}s")
        
        # Benchmark KD-Tree with exact median
        for method in ["variance", "interleave"]:
            tracemalloc.start()
            t0 = time.perf_counter()
            tree = KDTree2D(points, split_method=method, approx_median=False)
            build_time = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            t1 = time.perf_counter()
            for q in queries:
                tree.nearest(tuple(q))
            query_time = time.perf_counter() - t1

            results.append({
                "N": N,
                "method": method,
                "approx_median": False,
                "sample_size": 0,
                "build_time_s": build_time,
                "query_time_s": query_time,
                "peak_MB": peak / 1e6,
            })
            print(f"  {method:9} (exact):       build={build_time:.4f}s  query={query_time:.4f}s")
        
        # Benchmark KD-Tree with approximate median
        for method in ["variance", "interleave"]:
            tracemalloc.start()
            t0 = time.perf_counter()
            tree = KDTree2D(points, split_method=method, approx_median=True, sample_size=sample_size)
            build_time = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            t1 = time.perf_counter()
            for q in queries:
                tree.nearest(tuple(q))
            query_time = time.perf_counter() - t1

            results.append({
                "N": N,
                "method": method,
                "approx_median": True,
                "sample_size": sample_size,
                "build_time_s": build_time,
                "query_time_s": query_time,
                "peak_MB": peak / 1e6,
            })
            print(f"  {method:9} (approx-{sample_size:3}): build={build_time:.4f}s  query={query_time:.4f}s")

    df = pd.DataFrame(results)
    df.to_csv("./output/kdtree2d_approx_median_benchmark.csv", index=False)
    print("\n" + "="*80)
    print("✅ Results saved to ./output/kdtree2d_approx_median_benchmark.csv")
    print("="*80)

    # -------- Plot results --------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    linear_data = df[df["method"] == "linear"]
    
    #1: Build Time Comparison
    ax = axes[0]
    ax.plot(linear_data["N"], linear_data["build_time_s"], "-s", label="linear", linewidth=2, markersize=8)
    
    for method in ["variance", "interleave"]:
        # Exact median
        exact_data = df[(df["method"] == method) & (df["approx_median"] == False)]
        ax.plot(exact_data["N"], exact_data["build_time_s"], "-o", label=f"{method} (exact)", linewidth=2)
        
        # Approximate median
        approx_data = df[(df["method"] == method) & (df["approx_median"] == True)]
        ax.plot(approx_data["N"], approx_data["build_time_s"], "--^", 
               label=f"{method} (approx)", alpha=0.7, linewidth=2)
    
    ax.set_xlabel("N (points)", fontsize=11)
    ax.set_ylabel("Build Time (s)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_title("Build Time: Exact vs Approximate Median", fontsize=12, fontweight='bold')
    
    # 2: Query Time Comparison
    ax = axes[1]
    ax.plot(linear_data["N"], linear_data["query_time_s"], "-s", label="linear", linewidth=2, markersize=8)
    
    for method in ["variance", "interleave"]:
        exact_data = df[(df["method"] == method) & (df["approx_median"] == False)]
        ax.plot(exact_data["N"], exact_data["query_time_s"], "-o", label=f"{method} (exact)", linewidth=2)
        
        approx_data = df[(df["method"] == method) & (df["approx_median"] == True)]
        ax.plot(approx_data["N"], approx_data["query_time_s"], "--^",
               label=f"{method} (approx)", alpha=0.7, linewidth=2)
    
    ax.set_xlabel("N (points)", fontsize=11)
    ax.set_ylabel("Query Time (s)", fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_title("Query Time: Exact vs Approximate Median (log scale)", fontsize=12, fontweight='bold')
    
    # 3: Memory Usage
    ax = axes[2]
    ax.plot(linear_data["N"], linear_data["peak_MB"], "-s", label="linear", linewidth=2, markersize=8)
    
    for method in ["variance", "interleave"]:
        exact_data = df[(df["method"] == method) & (df["approx_median"] == False)]
        ax.plot(exact_data["N"], exact_data["peak_MB"], "-o", label=f"{method} (exact)", linewidth=2)
        
        approx_data = df[(df["method"] == method) & (df["approx_median"] == True)]
        ax.plot(approx_data["N"], approx_data["peak_MB"], "--^", label=f"{method} (approx)", alpha=0.7, linewidth=2)
    
    ax.set_xlabel("N (points)", fontsize=11)
    ax.set_ylabel("Peak Memory (MB)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title("Memory Usage Comparison", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("./output/kdtree2d_approx_median_benchmark.png", dpi=150)
    print("✅ Plots saved to ./output/kdtree2d_approx_median_benchmark.png")
    plt.show()


if __name__ == "__main__":
    benchmark_kdtree()