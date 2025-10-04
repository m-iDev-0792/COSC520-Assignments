import uuid
import sys
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Value, Lock
import time
import threading
import glob
import random

# ---------- Global Progress Helper ----------

progress_counter = None
progress_lock = None


def init_worker(counter, lock):
    """Initialize global variables in each worker process."""
    global progress_counter, progress_lock
    progress_counter = counter
    progress_lock = lock


def progress_updater(total_count):
    """Thread that updates a global tqdm bar."""
    with tqdm(total=total_count, desc="Overall Progress", unit="ID") as pbar:
        last_val = 0
        while True:
            with progress_lock:
                val = progress_counter.value
            diff = val - last_val
            if diff > 0:
                pbar.update(diff)
                last_val = val
            if val >= total_count:
                break
            time.sleep(0.5)


# ---------- Worker Function ----------

def generate_user_ids_chunk(start_index, count, output_dir, chunk_index, batch_size=100):
    filename = os.path.join(output_dir, f"user_ids_{chunk_index:05d}.txt")
    written = 0

    with open(filename, "w") as f:
        buffer = []
        for i in range(count):
            buffer.append(str(uuid.uuid4()) + "\n")

            # flush every batch_size or at the end
            if len(buffer) >= batch_size or i == count - 1:
                f.writelines(buffer)
                buffer.clear()

                # update global progress in batches
                with progress_lock:
                    progress_counter.value += min(batch_size, count - i + batch_size - 1)

                written += batch_size
    return filename


# ---------- Main Split + Generate ----------

def split_and_generate(total_count, chunk_size, output_dir, processes, counter, lock):
    global progress_counter, progress_lock
    progress_counter = counter
    progress_lock = lock

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_chunks = (total_count + chunk_size - 1) // chunk_size
    tasks = []

    for i in range(num_chunks):
        count = min(chunk_size, total_count - i * chunk_size)
        tasks.append((i * chunk_size, count, output_dir, i))

    print(f"Splitting {total_count:,} IDs into {num_chunks} chunks of up to {chunk_size:,} each...")

    # start progress updater thread
    t = threading.Thread(target=progress_updater, args=(total_count,), daemon=True)
    t.start()

    with Pool(processes=processes, initializer=init_worker, initargs=(counter, lock)) as pool:
        results = pool.starmap(generate_user_ids_chunk, tasks)

    t.join()  # wait for progress thread to finish

    print("\nGeneration complete. Files created:")
    for f in results:
        print("  -", f)


# ---------- Loader Function ----------
def load_user_ids_from_dir(directory, limit=None, shuffle=False):
    """
    Load up to `limit` user IDs from all dataset files in `directory`.

    Args:
        directory (str): Path to the dataset directory (e.g. "dataset").
        limit (int, optional): Maximum number of IDs to load. If None, load all.
        shuffle (bool, optional): If True, shuffle the file order before loading
                                  and also shuffle within loaded results.

    Returns:
        list[str]: Loaded user IDs.
    """
    # Find all dataset chunk files
    pattern = os.path.join(directory, "user_ids_*.txt")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No dataset files found in directory '{directory}'")

    if shuffle:
        random.shuffle(files)

    loaded = []
    total_loaded = 0

    for fname in files:
        with open(fname, "r") as f:
            for line in f:
                loaded.append(line.strip())
                total_loaded += 1
                if limit is not None and total_loaded >= limit:
                    if shuffle:
                        random.shuffle(loaded)
                    return loaded

    if shuffle:
        random.shuffle(loaded)
    return loaded


def load_user_ids_from_files(filenames, limit=None):
    """
    Load up to `limit` user IDs from multiple dataset files.
    If limit is None, load all.
    """
    loaded = []
    total_loaded = 0

    for fname in filenames:
        with open(fname, "r") as f:
            for line in f:
                loaded.append(line.strip())
                total_loaded += 1
                if limit is not None and total_loaded >= limit:
                    return loaded
    return loaded


#estimate memory if all IDs were loaded into a python list.
def estimate_memory_usage(total_count):
    sample_uid = str(uuid.uuid4())
    size_per_uid = sys.getsizeof(sample_uid)
    list_overhead = sys.getsizeof([])

    total_strings_size = size_per_uid * total_count
    total_list_size = list_overhead + total_strings_size

    print("\n" + "=" * 40)
    print("MEMORY USAGE ESTIMATE (if loaded into RAM)")
    print("=" * 40)
    print(f"Number of user IDs: {total_count:,}")
    print(f"One UUID string size: {size_per_uid:,} bytes")
    print(f"Strings total: {total_strings_size / 1024 / 1024:,.2f} MB")
    print(f"List overhead: {list_overhead:,} bytes")
    print(f"Total estimated: {total_list_size / 1024 / 1024 / 1024:,.2f} GB")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate unique user IDs in parallel with global progress, split into multiple files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python dataset_gen.py 1000000000 --chunk-size 1000000 --processes 8 -o dataset"
    )

    parser.add_argument("count", type=int, help="Total number of user IDs to generate")
    parser.add_argument("-c", "--chunk-size", type=int, default=1000000,
                        help="Number of IDs per file (default: 1,000,000)")
    parser.add_argument("-p", "--processes", type=int, default=cpu_count(),
                        help="Number of parallel processes (default: CPU count)")
    parser.add_argument("-o", "--output", type=str, default="dataset",
                        help="Output directory (default: dataset)")

    args = parser.parse_args()

    if args.count <= 0:
        parser.error("Count must be a positive number.")
    if args.chunk_size <= 0:
        parser.error("Chunk size must be a positive number.")

    #setup shared counter
    progress_counter = Value('i', 0)
    progress_lock = Lock()

    estimate_memory_usage(args.count)
    split_and_generate(args.count, args.chunk_size, args.output, args.processes, progress_counter, progress_lock)