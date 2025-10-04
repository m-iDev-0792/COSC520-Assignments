import time
from typing import List, Set
from tqdm import tqdm

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Load usernames from a text file (one username per line).
def load_usernames_from_file(filename: str) -> List[str]:
    with open(filename, 'r') as f:
        usernames = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(usernames):,} usernames from '{filename}'")
    return usernames

# Sort usernames alphabetically (required for binary search).
def sort_usernames(usernames: List[str]) -> List[str]:
    print(f"Sorting {len(usernames):,} usernames...")
    sorted_list = sorted(usernames)
    print("Sorting complete")
    return sorted_list

# Create a hash table from a list of usernames for O(1) lookup.
def create_username_hash_table(usernames: List[str]) -> Set[str]:
    print(f"Creating hash table with {len(usernames):,} usernames...")
    hash_table = set(usernames)
    print("Hash table created")
    return hash_table


# ============================================================================
# USERNAME CHECKER FUNCTIONS
# ============================================================================

def check_username_linear(username: str, usernames: List[str]) -> bool:
    """
    Check if username exists using linear search.
    Time Complexity: O(n)

    Args:
        username: Username to search for
        usernames: List of usernames to search in

    Returns:
        True if username exists, False otherwise
    """
    for user in usernames:
        if user == username:
            return True
    return False


def check_username_binary(username: str, sorted_usernames: List[str]) -> bool:
    """
    Check if username exists using binary search.
    Time Complexity: O(log n)
    Note: Requires sorted list of usernames

    Args:
        username: Username to search for
        sorted_usernames: Sorted list of usernames to search in

    Returns:
        True if username exists, False otherwise
    """
    left, right = 0, len(sorted_usernames) - 1

    # do binary search
    while left <= right:
        mid = (left + right) // 2
        mid_username = sorted_usernames[mid]

        if mid_username == username:
            return True
        elif mid_username < username:
            left = mid + 1
        else:
            right = mid - 1
    return False


def check_username_hash(username: str, username_set: Set[str]) -> bool:
    """
    Check if username exists using hash table lookup.
    Time Complexity: O(1) average case

    Args:
        username: Username to search for
        username_set: Set of usernames to search in

    Returns:
        True if username exists, False otherwise
    """
    return username in username_set



from typing import List, Set

class LinearChecker:
    """Username checker using linear search (O(n))."""

    def __init__(self, usernames: List[str]):
        self.usernames: List[str] = usernames
        self.dataset_size = len(usernames)
        print(f"LinearChecker initialized with {len(self.usernames):,} usernames")

    def contains(self, username: str) -> bool:
        for user in self.usernames:
            if user == username:
                return True
        return False
        
    def check(self, items):
        default_pair = ("", False)
        ret = [default_pair] * len(items)
        with tqdm(total=len(items), desc="Checking usernames") as pbar:
            for index, item in enumerate(items):
                if isinstance(item, tuple):
                    ret[index] = (item[0], self.contains(item[0]))
                else:
                        ret[index] = (item, self.contains(item))
                pbar.update(1)
        return ret


class BinaryChecker:
    """Username checker using binary search (O(log n))."""

    def __init__(self, usernames: List[str]):
        print(f"Sorting {len(usernames):,} usernames for BinaryChecker...")
        self.sorted_usernames: List[str] = sorted(usernames)
        self.dataset_size = len(usernames)
        print("BinaryChecker initialized")

    def contains(self, username: str) -> bool:
        left, right = 0, len(self.sorted_usernames) - 1
        while left <= right:
            mid = (left + right) // 2
            mid_username = self.sorted_usernames[mid]

            if mid_username == username:
                return True
            elif mid_username < username:
                left = mid + 1
            else:
                right = mid - 1
        return False

    def check(self, items):
        default_pair = ("", False)
        ret = [default_pair] * len(items)
        with tqdm(total=len(items), desc="Checking usernames") as pbar:
            for index, item in enumerate(items):
                if isinstance(item, tuple):
                    ret[index] = (item[0], self.contains(item[0]))
                else:
                    ret[index] = (item, self.contains(item))
                pbar.update(1)
        return ret

class HashChecker:
    """Username checker using hash table lookup (O(1) avg)."""

    def __init__(self, usernames: List[str]):
        print(f"Creating hash set for {len(usernames):,} usernames...")
        self.username_set: Set[str] = set(usernames)
        self.dataset_size = len(usernames)
        print("HashChecker initialized")

    def contains(self, username: str) -> bool:
        return username in self.username_set

    def check(self, items):
        default_pair = ("", False)
        ret = [default_pair] * len(items)
        with tqdm(total=len(items), desc="Checking usernames") as pbar:
            for index, item in enumerate(items):
                if isinstance(item, tuple):
                    ret[index] = (item[0], self.contains(item[0]))
                else:
                    ret[index] = (item, self.contains(item))
                pbar.update(1)
        return ret