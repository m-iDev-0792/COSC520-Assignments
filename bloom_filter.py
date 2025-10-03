import hashlib
import math

class BloomFilter:
    def __init__(self, n, p):
        """
        n: expected number of elements
        p: target false positive rate
        """
        #optimal size of the bit array
        self.m = int(- (n * math.log(p)) / (math.log(2) ** 2))
        # Optimal number of hash functions
        self.k = int((self.m / n) * math.log(2))
        #bit array
        self.bit_array = [0] * self.m

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

