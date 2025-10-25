"""
Generate N random 2D points inside a rectangle and write them to text files.

Defaults:
- Rectangle: (x_min, y_min) = (0, 0), (x_max, y_max) = (500, 500)
- Output folder: ./dataset
- File format: plain text, one "x y" pair per line
- Parallel generation with --num-workers workers
"""

from __future__ import annotations
import argparse
import math
import multiprocessing as mp
from pathlib import Path
import numpy as np
from typing import Tuple, List
from tqdm import tqdm


def _gen_and_write_chunk(args: Tuple[int, int, float, float, float, float, Path, int, int, int]):
    """
    Worker function to generate a chunk of points and write to a single text file.
    """
    (chunk_idx, points_in_chunk, x_min, x_max, y_min, y_max, out_dir,
     base_seed, chunk_seed_stride, float_precision) = args

    # Independent RNG per chunk for reproducibility
    seed = (base_seed + chunk_idx * chunk_seed_stride) % (2**32 - 1)
    rng = np.random.default_rng(seed)

    # Use vectorized generation
    xs = rng.uniform(x_min, x_max, size=points_in_chunk)
    ys = rng.uniform(y_min, y_max, size=points_in_chunk)

    # Choose dtype based on precision
    if float_precision == 32:
        xs = xs.astype(np.float32, copy=False)
        ys = ys.astype(np.float32, copy=False)
        fmt = "%.7g %.7g\n"
    else:
        fmt = "%.15g %.15g\n"

    # Write text file
    fname = out_dir / f"chunk_{chunk_idx:06d}.txt"
    with fname.open("w", buffering=1024 * 1024) as f:
        # Interleave columns without building a huge (N,2) array in memory
        for x, y in zip(xs, ys):
            f.write(fmt % (x, y))

    return str(fname)


def generate_points(
    n_points: int,
    rect: Tuple[float, float, float, float],
    out_dir: Path,
    chunk_size: int,
    num_workers: int,
    base_seed: int,
    overwrite: bool,
    float_precision: int,
) -> List[Path]:
    x_min, y_min, x_max, y_max = rect
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid rectangle: max must be greater than min.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(out_dir.glob("*.txt"))
    if existing_files and not overwrite:
        raise FileExistsError(
            f"Output directory '{out_dir}' already contains text files. "
            f"Use --overwrite to remove them first or choose a different folder."
        )
    if existing_files and overwrite:
        for p in existing_files:
            p.unlink()

    # Work plan
    n_chunks = math.ceil(n_points / chunk_size)
    chunk_counts = [chunk_size] * n_chunks
    remainder = n_points - chunk_size * (n_chunks - 1)
    if n_chunks > 0:
        chunk_counts[-1] = remainder
    print(f'Generating {n_points} points, the task is divided into {n_chunks} sub-tasks')
    # Prepare parallel jobs
    jobs = [
        (idx, cnt, x_min, x_max, y_min, y_max, out_dir, base_seed, 9973, float_precision)
        for idx, cnt in enumerate(chunk_counts)
    ]

    written = []
    
    # Progress bar for chunks
    with tqdm(total=n_chunks, desc="Generating chunks", unit="chunk") as pbar:
        if num_workers <= 1:
            for job in jobs:
                _gen_and_write_chunk(job)
                written.append(out_dir / f"chunk_{job[0]:06d}.txt")
                pbar.update(1)
        else:
            # parallel generation
            with mp.Pool(processes=num_workers) as pool:
                for path_str in pool.imap_unordered(_gen_and_write_chunk, jobs, chunksize=1):
                    written.append(Path(path_str))
                    pbar.update(1)

    #sort by chunk index
    written.sort()
    return written


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="Total number of points to generate (e.g., 10000000).")
    ap.add_argument("--rect", type=float, nargs=4, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
                    default=(0.0, 0.0, 500.0, 500.0),
                    help="Rectangle bounds.")
    ap.add_argument("--out", type=Path, default=Path("./dataset"), help="Output folder.")
    ap.add_argument("--chunk-size", type=int, default=1_000_000,
                    help="Number of points per file. Tune to balance RAM, CPU, and I/O.")
    ap.add_argument("--num-workers", type=int, default=max(mp.cpu_count() - 1, 1),
                    help="Parallel workers (processes).")
    ap.add_argument("--seed", type=int, default=12345, help="Base RNG seed for reproducibility.")
    ap.add_argument("--overwrite", action="store_true", help="Delete existing .txt files in output folder first.")
    ap.add_argument("--float-precision", type=int, choices=(32, 64), default=32,
                    help="Precision used for output values (text still; this controls digits/rounding).")
    return ap.parse_args()


def main():
    args = parse_args()
    written = generate_points(
        n_points=args.n,
        rect=tuple(args.rect),
        out_dir=args.out,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        base_seed=args.seed,
        overwrite=args.overwrite,
        float_precision=args.float_precision,
    )
    print(f"\nWrote {len(written)} files to {args.out.resolve()}")
    total_lines = sum(int(p.stat().st_size > 0) for p in written)  # lightweight check
    for p in written:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()