import hashlib
import math
from tqdm import tqdm
class BloomFilter:
    def __init__(self, n, p, usernames = None):
        """
        n: expected number of elements
        p: target false positive rate
        usernames: list of usernames
        """
        #optimal size of the bit array
        self.m = int(- (n * math.log(p)) / (math.log(2) ** 2))
        # Optimal number of hash functions
        self.k = int((self.m / n) * math.log(2))
        #bit array
        self.bit_array = [0] * self.m
        self.dataset_size = n
        if usernames is not None:
            for username in usernames:
                if isinstance(username, tuple):
                    self.add(username[0])
                else:
                    self.add(username)

    def _hashes(self, item):
        # generate k different hash values using different seeds
        for i in range(self.k):
            h = int(hashlib.md5((str(i) + item).encode()).hexdigest(), 16)
            yield h % self.m

    def add(self, item):
        #insert an element into the filter
        for pos in self._hashes(item):
            self.bit_array[pos] = 1

    def contains(self, item):
        return all(self.bit_array[pos] == 1 for pos in self._hashes(item))

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