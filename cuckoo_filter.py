import hashlib
import random

class CuckooFilter:
    def __init__(self, capacity=1000, bucket_size=4, fingerprint_size=8, max_kicks=500):
        """
        capacity: expected number of elements
        bucket_size: number of slots per bucket
        fingerprint_size: fingerprint size in bits
        max_kicks: maximum number of displacements before giving up
        """
        self.bucket_size = bucket_size
        self.fingerprint_size = fingerprint_size
        self.max_kicks = max_kicks
        self.num_buckets = capacity // bucket_size
        self.buckets = [[] for _ in range(self.num_buckets)]

    def _fingerprint(self, item):
        h = hashlib.md5(item.encode()).hexdigest()
        fp = int(h, 16) % (1 << self.fingerprint_size)
        return fp if fp != 0 else 1  # avoid 0 as invalid value

    def _index_hash(self, item):
        return int(hashlib.sha1(item.encode()).hexdigest(), 16) % self.num_buckets

    def _alt_index(self, index, fp):
        h = int(hashlib.sha1(str(fp).encode()).hexdigest(), 16)
        return (index ^ (h % self.num_buckets)) % self.num_buckets

    def add(self, item):
        fp = self._fingerprint(item)
        i1 = self._index_hash(item)
        i2 = self._alt_index(i1, fp)

        #try to insert into bucket i1 or i2
        for i in (i1, i2):
            if len(self.buckets[i]) < self.bucket_size:
                self.buckets[i].append(fp)
                return True

        #relocation (cuckoo kick-out process)
        i = random.choice([i1, i2])
        for _ in range(self.max_kicks):
            j = random.randrange(len(self.buckets[i]))
            self.buckets[i][j], fp = fp, self.buckets[i][j]  # swap
            i = self._alt_index(i, fp)
            if len(self.buckets[i]) < self.bucket_size:
                self.buckets[i].append(fp)
                return True
        return False  # insertion failed (filter too full)

    def contains(self, item):
        fp = self._fingerprint(item)
        i1 = self._index_hash(item)
        i2 = self._alt_index(i1, fp)
        return fp in self.buckets[i1] or fp in self.buckets[i2]

    def delete(self, item):
        fp = self._fingerprint(item)
        i1 = self._index_hash(item)
        i2 = self._alt_index(i1, fp)
        for i in (i1, i2):
            if fp in self.buckets[i]:
                self.buckets[i].remove(fp)
                return True
        return False
