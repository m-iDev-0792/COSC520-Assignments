
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import time
import tracemalloc
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class QuadNode:
    bounds: Tuple[float,float,float,float]   # (minx,miny,maxx,maxy)
    point_idx: Optional[int] = None          # store at most one point per node (leaf) before split
    children: Optional[Tuple[int,int,int,int]] = None  # indices into nodes list: (SW, SE, NW, NE)

class QuadTree2D:
    """
    Point Quad-Tree with capacity=1 and recursive subdivision.
    """
    def __init__(self, points: np.ndarray, capacity: int = 1):
        assert points.ndim == 2 and points.shape[1] == 2, "points must be Nx2"
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.capacity = int(capacity)
        self.nodes: List[QuadNode] = []

        if points.shape[0] == 0:
            self.root = -1
            return

        minx = float(points[:,0].min())
        miny = float(points[:,1].min())
        maxx = float(points[:,0].max())
        maxy = float(points[:,1].max())
        # pad tiny epsilon to avoid degenerate boxes
        eps = 1e-12
        root_bounds = (minx-eps, miny-eps, maxx+eps, maxy+eps)
        self.root = self._make_node(root_bounds)

        for i in range(points.shape[0]):
            self._insert(self.root, i)

    # --------------- building helpers ---------------
    def _make_node(self, bounds) -> int:
        nid = len(self.nodes)
        self.nodes.append(QuadNode(bounds=bounds))
        return nid

    def _subdivide(self, nid: int):
        minx,miny,maxx,maxy = self.nodes[nid].bounds
        cx = (minx+maxx)/2.0
        cy = (miny+maxy)/2.0
        sw = self._make_node((minx, miny, cx,   cy))
        se = self._make_node((cx,   miny, maxx, cy))
        nw = self._make_node((minx, cy,   cx,   maxy))
        ne = self._make_node((cx,   cy,   maxx, maxy))
        self.nodes[nid].children = (sw, se, nw, ne)

    def _child_for_point(self, nid: int, px: float, py: float) -> int:
        minx,miny,maxx,maxy = self.nodes[nid].bounds
        cx = (minx+maxx)/2.0
        cy = (miny+maxy)/2.0
        east = px >= cx
        north = py >= cy
        sw,se,nw,ne = self.nodes[nid].children  # type: ignore
        if not east and not north: return sw
        if east and not north:     return se
        if not east and north:     return nw
        return ne

    def _insert(self, nid: int, pid: int):
        node = self.nodes[nid]
        if node.children is None:
            # leaf
            if node.point_idx is None:
                node.point_idx = pid
                return
            else:
                # split
                old_pid = node.point_idx
                node.point_idx = None
                self._subdivide(nid)
                self._insert(self._child_for_point(nid, *self.points[old_pid]), old_pid)  # reinsert
                self._insert(self._child_for_point(nid, *self.points[pid]), pid)
        else:
            self._insert(self._child_for_point(nid, *self.points[pid]), pid)

    # --------------- queries
    def range_query(self, rect: Tuple[float,float,float,float]) -> List[int]:
        if self.root == -1: return []
        rx0, ry0, rx1, ry1 = rect
        def overlap(a,b):
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)
        out: List[int] = []
        stack = [self.root]
        while stack:
            nid = stack.pop()
            node = self.nodes[nid]
            if not overlap(node.bounds, rect):
                continue
            if node.point_idx is not None:
                px, py = self.points[node.point_idx]
                if rx0 <= px <= rx1 and ry0 <= py <= ry1:
                    out.append(int(node.point_idx))
            if node.children is not None:
                stack.extend(node.children)
        return out

    def nearest(self, target: Tuple[float,float]) -> Tuple[int, float]:
        if self.root == -1:
            return -1, float("inf")
        tx, ty = target
        best_idx = -1
        best_d2 = float("inf")
        def box_d2(b):
            minx,miny,maxx,maxy = b
            dx = 0.0 if minx <= tx <= maxx else (minx-tx if tx < minx else tx-maxx)
            dy = 0.0 if miny <= ty <= maxy else (miny-ty if ty < miny else ty-maxy)
            return dx*dx + dy*dy

        stack = [self.root]
        while stack:
            nid = stack.pop()
            node = self.nodes[nid]
            if box_d2(node.bounds) >= best_d2:
                continue
            if node.point_idx is not None:
                px, py = self.points[node.point_idx]
                d2 = (px-tx)*(px-tx) + (py-ty)*(py-ty)
                if d2 < best_d2:
                    best_d2 = d2
                    best_idx = int(node.point_idx)
            if node.children is not None:
                # explore children sorted by distance to bbox (rough best-first)
                ch = list(node.children)
                ch.sort(key=lambda c: box_d2(self.nodes[c].bounds), reverse=True)
                # push far first so near is processed next
                for c in ch:
                    stack.append(c)
        return best_idx, float(best_d2)


# ---------------------- Benchmark----------------------
def _benchmark_impl():
    rng = np.random.default_rng(123)
    Ns = [10_000, 50_000, 100_000]
    results = []
    for N in Ns:
        pts = rng.random((N,2))
        queries = rng.random((1000,2))
        tracemalloc.start()
        t0 = time.perf_counter()
        qt = QuadTree2D(pts)
        build = time.perf_counter()-t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        t1 = time.perf_counter()
        for q in queries:
            qt.nearest((float(q[0]), float(q[1])))
        query = time.perf_counter()-t1

        results.append({"N": N, "build_time_s": build, "query_time_s": query, "peak_MB": peak/1e6})
        print(f"N={N:7d}  build={build:.4f}s  query={query:.4f}s  mem={peak/1e6:.2f}MB")

    df = pd.DataFrame(results)
    Path("./output").mkdir(exist_ok=True, parents=True)
    df.to_csv("./output/quadtree2d_benchmark.csv", index=False)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df["N"], df["build_time_s"], marker='o', label="build")
    ax.plot(df["N"], df["query_time_s"], marker='s', label="query (1000 nn)")
    ax.set_xlabel("N"); ax.set_ylabel("Time (s)"); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("./output/quadtree2d_benchmark.png", dpi=150)

if __name__ == "__main__":
    _benchmark_impl()
