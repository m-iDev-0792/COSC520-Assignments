
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal
import numpy as np
import pandas as pd
import time
import tracemalloc
import matplotlib.pyplot as plt

SplitMethod = Literal["variance", "interleave"]

@dataclass
class _NodeView:
    """A lightweight view into an implicit node range [lo, hi) within the index array."""
    idx_lo: int
    idx_hi: int
    node_id: int
    axis: int
    bounds: Tuple[float, float, float, float]  # (minx, miny, maxx, maxy)


class ImplicitKDTree2D:
    """
    2D KD-Tree with implicit array representation (heap-like, static).
    - No explicit Node objects; the tree is stored in arrays.
    - Construction supports variance-based or interleaving axis selection.
    - Query: nearest neighbor and axis-aligned rectangle range query.
    """
    def __init__(self, points: np.ndarray, split_method: SplitMethod = "variance",
                 approx_median: bool = False, sample_size: int = 256, seed: Optional[int] = None):
        assert points.ndim == 2 and points.shape[1] == 2, "points must be Nx2"
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        N = self.points.shape[0]
        if seed is not None:
            np.random.seed(seed)

        self.split_method = split_method
        self.approx_median = approx_median
        self.sample_size = sample_size

        # permutation of point indices; during build, subarrays are partitioned in-place
        self.perm = np.arange(N, dtype=np.int64)

        # implicit arrays (heap-like) sized to next power-of-two - 1 upper bound
        # but we will grow dynamically as we assign nodes
        self.axis = []             # axis per node_id (0 or 1)
        self.split_idx = []        # index into perm (global) of median point for the node
        self.bounds = []           # (minx,miny,maxx,maxy) per node_id

        # Build
        if N == 0:
            self.root_id = -1
            return

        # global bounds
        minx = float(self.points[:,0].min())
        miny = float(self.points[:,1].min())
        maxx = float(self.points[:,0].max())
        maxy = float(self.points[:,1].max())
        self.root_id = 0
        # reserve root
        self._ensure_node_capacity(0)
        self._build(0, 0, N, depth=0, bounds=(minx, miny, maxx, maxy))

    # internal utils
    def _ensure_node_capacity(self, node_id: int):
        need = node_id + 1
        while len(self.axis) < need:
            self.axis.append(-1)
            self.split_idx.append(-1)
            self.bounds.append((0.0,0.0,0.0,0.0))

    def _choose_axis(self, lo: int, hi: int, depth: int) -> int:
        if self.split_method == "interleave":
            return depth & 1
        # variance-based
        pts = self.points[self.perm[lo:hi]]
        v0 = np.var(pts[:,0])
        v1 = np.var(pts[:,1])
        return 0 if v0 >= v1 else 1

    def _median_partition(self, lo: int, hi: int, axis: int) -> int:
        """Rearrange self.perm[lo:hi] so median element by coordinate is at m; return m index (global)."""
        m = (lo + hi) // 2
        # Approximate median via sampling
        if self.approx_median:
            size = min(self.sample_size, hi - lo)
            sample_idxs = np.random.choice(np.arange(lo, hi), size=size, replace=False)
            sample_coords = self.points[self.perm[sample_idxs], axis]
            approx_val = np.median(sample_coords)
            # three-way partition around approx_val and put pivot near middle
            coords = self.points[self.perm[lo:hi], axis]
            local = np.arange(lo, hi)
            left = local[coords < approx_val]
            right = local[coords > approx_val]
            eq = local[(coords == approx_val)]
            # construct order and place middle as m
            order = np.concatenate([left, eq, right])
            self.perm[lo:hi] = self.perm[order]
            # fall back to nth selection for exact median pos
        # ensure exact median at m
        coords = self.points[self.perm[lo:hi], axis]
        # argpartition produces median at relative index k
        k = m - lo
        order = np.argpartition(coords, k)
        self.perm[lo:hi] = self.perm[lo:hi][order]
        return m

    def _child(self, node_id: int, left: bool) -> int:
        # heap children
        return 2*node_id + (1 if left else 2)

    def _build(self, node_id: int, lo: int, hi: int, depth: int, bounds: Tuple[float,float,float,float]):
        self._ensure_node_capacity(node_id)
        self.bounds[node_id] = bounds

        if hi - lo <= 0:
            self.axis[node_id] = -1
            self.split_idx[node_id] = -1
            return

        axis = self._choose_axis(lo, hi, depth)
        m = self._median_partition(lo, hi, axis)
        self.axis[node_id] = axis
        self.split_idx[node_id] = m

        px, py = self.points[self.perm[m]]
        minx,miny,maxx,maxy = bounds
        if axis == 0:
            left_bounds  = (minx, miny, px,   maxy)
            right_bounds = (px,   miny, maxx, maxy)
        else:
            left_bounds  = (minx, miny, maxx, py)
            right_bounds = (minx, py,   maxx, maxy)

        # build children
        li = self._child(node_id, True)
        ri = self._child(node_id, False)
        if lo < m:
            self._build(li, lo, m, depth+1, left_bounds)
        if m+1 < hi:
            self._build(ri, m+1, hi, depth+1, right_bounds)

    # queries
    def nearest(self, target: Tuple[float,float]) -> Tuple[int, float]:
        """Return (point_index, squared_distance)."""
        if self.root_id == -1:
            return -1, float("inf")
        best_idx = -1
        best_d2 = float("inf")
        tx, ty = target

        def box_dist2(b):
            minx,miny,maxx,maxy = b
            dx = 0.0 if minx <= tx <= maxx else (minx-tx if tx < minx else tx-maxx)
            dy = 0.0 if miny <= ty <= maxy else (miny-ty if ty < miny else ty-maxy)
            return dx*dx + dy*dy

        stack = [self.root_id]
        while stack:
            nid = stack.pop()
            if nid >= len(self.axis) or self.axis[nid] == -1:
                continue
            # prune by bbox
            if box_dist2(self.bounds[nid]) >= best_d2:
                continue
            m = self.split_idx[nid]
            pid = self.perm[m]
            px, py = self.points[pid]
            d2 = (px-tx)*(px-tx) + (py-ty)*(py-ty)
            if d2 < best_d2:
                best_d2 = d2
                best_idx = pid

            axis = self.axis[nid]
            go_left_first = (tx < px) if axis == 0 else (ty < py)

            li = self._child(nid, True)
            ri = self._child(nid, False)
            first, second = (li, ri) if go_left_first else (ri, li)
            # visit nearer child first
            stack.append(second)
            stack.append(first)

        return best_idx, float(best_d2)

    def range_query(self, rect: Tuple[float,float,float,float]) -> List[int]:
        """Return indices of points within axis-aligned rect (minx,miny,maxx,maxy)."""
        if self.root_id == -1:
            return []
        rx0, ry0, rx1, ry1 = rect
        results: List[int] = []
        def box_overlap(a, b):
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

        stack = [self.root_id]
        while stack:
            nid = stack.pop()
            if nid >= len(self.axis) or self.axis[nid] == -1:
                continue
            if not box_overlap(self.bounds[nid], rect):
                continue
            m = self.split_idx[nid]
            pid = self.perm[m]
            px, py = self.points[pid]
            if rx0 <= px <= rx1 and ry0 <= py <= ry1:
                results.append(pid)
            li = self._child(nid, True)
            ri = self._child(nid, False)
            stack.append(ri)
            stack.append(li)
        return results


# ---------------------- Benchmark ----------------------

def _benchmark_impl():
    import numpy as np
    Ns = [10_000, 50_000, 100_000, 200_000]
    results = []
    rng = np.random.default_rng(42)
    for N in Ns:
        pts = rng.random((N,2))
        queries = rng.random((1000,2))
        for method in ["variance", "interleave"]:
            for approx, sample in [(False, 0), (True, 128)]:
                tracemalloc.start()
                t0 = time.perf_counter()
                tree = ImplicitKDTree2D(pts, split_method=method, approx_median=approx, sample_size=sample)
                build_time = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                t1 = time.perf_counter()
                for q in queries:
                    tree.nearest((float(q[0]), float(q[1])))
                query_time = time.perf_counter() - t1

                results.append({
                    "N": N, "method": method, "approx": approx, "sample": sample,
                    "build_time_s": build_time, "query_time_s": query_time, "peak_MB": peak/1e6
                })
                print(f"N={N:7d}  {method:10s} approx={approx:<5} build={build_time:.4f}s  query={query_time:.4f}s  mem={peak/1e6:.2f}MB")
    df = pd.DataFrame(results)
    out_csv = "./output/implicit_kdtree2d_benchmark.csv"
    Path("./output").mkdir(exist_ok=True, parents=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved metrics to {out_csv}")
    # simple plot
    fig, ax = plt.subplots(figsize=(6,4))
    for method in ["variance","interleave"]:
        xs = sorted(set(df[df.method==method]["N"]))
        ys = [df[(df.method==method)&(df.N==n)&(df.approx==False)]["build_time_s"].values[0] for n in xs]
        ax.plot(xs, ys, marker='o', label=f"{method}-build")
    ax.set_xlabel("N"); ax.set_ylabel("Build time (s)"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig("./output/implicit_kdtree2d_build.png", dpi=150)

if __name__ == "__main__":
    _benchmark_impl()
