import hashlib
import random
from tqdm import tqdm
import math


def cal_cuckoo_params(usernames, p, bucket_size=4, load_factor=0.95):
    """
    n: number of expected elements
    p: target false positive rate
    bucket_size: slots per bucket (default 4)
    load_factor: expected load factor (default 0.95)
    """
    n = len(usernames)
    # fingerprint size f (bits)
    f = math.ceil(math.log2(bucket_size / p))
    # number of buckets
    num_buckets = math.ceil(n / (bucket_size * load_factor))
    # capacity
    capacity = num_buckets * bucket_size
    return {
        "usernames": usernames,
        "dataset_size": n,
        "capacity": capacity,
        "bucket_size": bucket_size,
        "fingerprint_size": f,
        "max_kicks": 500
    }

class CuckooFilter:
    def __init__(self, dataset_size, capacity=1000, bucket_size=4, fingerprint_size=8, max_kicks=500, usernames=None):
        """
        usernames: list of usernames
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
        self.dataset_size = dataset_size
        self.capacity = capacity
        if usernames is not None:
            for username in usernames:
                if isinstance(username, tuple):
                    self.add(username[0])
                else:
                    self.add(username)

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