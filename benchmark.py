import os, sys
import time
from typing import List, Set
from simple_login_checker_algo import *

def benchmark_searches(usernames_to_check: List[str],
                       username_list: List[str],
                       sorted_list: List[str],
                       username_set: Set[str]) -> None:
    """
    Benchmark all three search methods.

    Args:
        usernames_to_check: List of usernames to search for
        username_list: Unsorted list for linear search
        sorted_list: Sorted list for binary search
        username_set: Hash set for hash table search
    """
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Testing {len(usernames_to_check)} username lookups...")
    print()

    # Linear Search
    start_time = time.time()
    linear_results = [check_username_linear(user, username_list)
                      for user in usernames_to_check]
    linear_time = time.time() - start_time

    # Binary Search
    start_time = time.time()
    binary_results = [check_username_binary(user, sorted_list)
                      for user in usernames_to_check]
    binary_time = time.time() - start_time

    # Hash Table Search
    start_time = time.time()
    hash_results = [check_username_hash(user, username_set)
                    for user in usernames_to_check]
    hash_time = time.time() - start_time

    # Display results
    print(f"Linear Search:     {linear_time:.6f} seconds | Found: {sum(linear_results)}")
    print(f"Binary Search:     {binary_time:.6f} seconds | Found: {sum(binary_results)}")
    print(f"Hash Table Search: {hash_time:.6f} seconds | Found: {sum(hash_results)}")
    print()
    print(f"Binary is {linear_time / binary_time:.2f}x faster than Linear")
    print(f"Hash Table is {linear_time / hash_time:.2f}x faster than Linear")
    print("=" * 70)


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    filename = "user_ids.txt"

    # Load usernames
    print("Loading usernames...")
    usernames = load_usernames_from_file(filename)

    if not usernames:
        print("No usernames loaded. Please ensure the file exists.")
        exit(1)

    # Prepare data structures for different search methods
    print("\nPreparing data structures...")
    sorted_usernames = sort_usernames(usernames)
    username_hash_set = create_username_hash_table(usernames)

    # Test searches
    print("\n" + "=" * 70)
    print("INDIVIDUAL SEARCH TESTS")
    print("=" * 70)

    # Test with existing username (first one)
    test_username_exists = usernames[0]
    print(f"\nSearching for existing username: '{test_username_exists}'")
    print(f"  Linear Search: {check_username_linear(test_username_exists, usernames)}")
    print(f"  Binary Search: {check_username_binary(test_username_exists, sorted_usernames)}")
    print(f"  Hash Search:   {check_username_hash(test_username_exists, username_hash_set)}")

    # Test with non-existing username
    test_username_not_exists = "nonexistent_user_xyz_12345"
    print(f"\nSearching for non-existing username: '{test_username_not_exists}'")
    print(f"  Linear Search: {check_username_linear(test_username_not_exists, usernames)}")
    print(f"  Binary Search: {check_username_binary(test_username_not_exists, sorted_usernames)}")
    print(f"  Hash Search:   {check_username_hash(test_username_not_exists, username_hash_set)}")

    # Benchmark with multiple searches
    print("\n" + "=" * 70)
    test_usernames = usernames[:100] if len(usernames) >= 100 else usernames[:10]
    # Add some non-existing usernames
    test_usernames.extend([f"fake_user_{i}" for i in range(10)])

    benchmark_searches(test_usernames, usernames, sorted_usernames, username_hash_set)