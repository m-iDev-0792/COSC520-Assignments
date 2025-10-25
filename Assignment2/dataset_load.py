"""
Load an arbitrary number of 2D points from a dataset folder of chunked text files.

- Each file is plain text: one line with "x y"
- Returns the first N points across files (optionally shuffles the result)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Optional
import numpy as np
import random
import tqdm


def iter_points_from_folder(folder: Path, shuffle_files: bool = False) -> Iterator[Tuple[float, float]]:
    """
    Stream points from all *.txt files in the folder
    If shuffle_files=True, randomize file order.
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.txt"))
    if shuffle_files:
        random.shuffle(files)

    bufsize = 1024 * 1024  # 1MB buffered reads
    for fpath in tqdm.tqdm(files, desc="Loading points from folder"):
        with fpath.open("r", buffering=bufsize) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Support space or comma separation
                if "," in line:
                    parts = line.split(",")
                else:
                    parts = line.split()
                if len(parts) < 2:
                    continue  # skip
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                except ValueError:
                    continue
                yield (x, y)


def load_points(
    folder: Path,
    n: Optional[int] = None,
    shuffle_result: bool = False,
    as_numpy: bool = True,
    dtype: str = "float32",
) -> np.ndarray | List[Tuple[float, float]]:
    """
    Load up to n points from folder. If n is None, load ALL available points.
    If shuffle_result is True, shuffle only the returned set (not the dataset on disk).
    """
    it = iter_points_from_folder(folder, shuffle_files=False)

    if n is None:
        data = list(it)
        if shuffle_result:
            random.shuffle(data)
        if as_numpy:
            return np.asarray(data, dtype=dtype)
        return data

    # Load exactly n points if possible
    data: List[Tuple[float, float]] = []
    for p in it:
        data.append(p)
        if len(data) >= n:
            break

    if len(data) < n:
        raise ValueError(
            f"Requested {n} points but only found {len(data)} in '{folder}'. "
            "Add more chunks or reduce n."
        )

    if shuffle_result:
        random.shuffle(data)

    if as_numpy:
        return np.asarray(data, dtype=dtype)
    return data


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--folder", type=Path, default=Path("./dataset"), help="Dataset folder containing *.txt chunks.")
    ap.add_argument("--n", type=int, default=1000, help="Number of points to load (use -1 for ALL).")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle the returned points.")
    ap.add_argument("--numpy", dest="as_numpy", action="store_true", help="Return a NumPy array.")
    ap.add_argument("--no-numpy", dest="as_numpy", action="store_false", help="Return a Python list of tuples.")
    ap.set_defaults(as_numpy=True)
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float32", help="Array dtype if returning NumPy.")
    return ap.parse_args()


def main():
    args = parse_args()
    n = None if args.n == -1 else args.n
    arr_or_list = load_points(
        folder=args.folder,
        n=n,
        shuffle_result=args.shuffle,
        as_numpy=args.as_numpy,
        dtype=args.dtype,
    )
    if isinstance(arr_or_list, np.ndarray):
        print(f"Loaded shape: {arr_or_list.shape}, dtype: {arr_or_list.dtype}")
    else:
        print(f"Loaded {len(arr_or_list)} points (list of tuples).")


if __name__ == "__main__":
    main()
