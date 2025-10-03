import os, sys
import time
from typing import List, Set
from simple_login_checker_algo import *
from bloom_filter import *
from cuckoo_filter import *
import random

usernames = ['riyn3u11', 'vftvhz', 'hyp7pwj7bxl', 'w015yjn', 'idlr706553', 'lhoq4zpqhro', 'ds1ef5ird83', 'bqy0hxc8s', 's4mokbp3', 't4st88v4woj7', 'l650b9ek5', 'b8aldkcr', 'ebuv5jpouq', 'k4lz7ypvb8h2', 'q15mt30n', 'c1g4vq', 'm33a3k', 'i1v3vue2lwg', 'a0lldz', 'v35azec', 'xm414rh9b', 'h80lxqx', 'eh1brgk24', 'l6dgbemnb1', 'uxjdt9', 'i917zey6xfn', 'j579cj', 'aa5mh5', 'fjv9cse', 'hoxbua37ktf', 'pentgrsfy', 'n7t25ebxl', 'e12nopfmvc', 'm683v0ao2', 'no54glvtyl7', 'fcg9bq7fybo', 'kmmap7m', 'ggc2u6', 'dwr25sjecd0g', 'pns4l62', 'phiefn3s', 'ifj0n1wc', 'upsca0q', 'qrk2aq', 'n5rfqbwu5', 'y4rehkim142y', 'dxplrben', 'gh6n7rug9i0', 'hv3i9fv56', 'lvwrdyjnn2', 'sumszv647s1', 'nqvnnn', 'y5aj8bb6', 'i6fywczjpb', 'qqkr1e', 'go1dxlsvxl', 'imixwp', 'z3ep5dq9', 'c3vtp9', 'ge6ojb', 'h02ftmqj36hu', 'cpzm8qz4sw', 'moljt7y', 'fx3wui', 'xeqx51crzlr6', 'd03j6h61o', 'jhfcl3okrtbc', 'nwrerj', 'xa4c19', 'tw69kk86y4vg', 'tnqd4o4acwi', 'm6i47db4zpe', 'owwjsgbkt7u', 'dlq4x2wl', 'wwmkn0v9z8', 'krnsn9hwv6', 'squ2fl2', 'kgz838lgx', 'es8g1pv68u7u', 'kvonxeaxr', 'faom1oe50', 'f6hbr81o', 'grvu9ojhrgq', 'bpl67u', 'gml9g654', 'xdefsjun', 'xyba8ffmn5v7', 'w2217cct', 'fmdl37pc', 'eavv5onawbd5', 'kwt1a1p', 'pyu0627irci', 'anwr77mdpgm', 'hr3ltrupwb0c', 'crikkya', 'zj8rx7zt', 'obwd0n1r0nj', 'bd9ruvnmyztx', 'hddyhmpmmh', 'dmuzaa26ab40']

username_queries = ['kvqdmy1lzk8d', 'fm99lbj1b', 'ggc2u6', 'i917zey6xfn', '17f40q7mcs6ubp', 'o7eb8yk9i69q', 's4mokbp3', 'fcg9bq7fybo', 'gml9g654', 'irj1dxa2', 't7t7dpqop0ip', 'h9bgmee', 'eavv5onawbd5', 'moljt7y', 'dmuzaa26ab40', 'koqsfamim', 'vyi83419hl', 'xa4c19', 'nqvnnn', 'dlq4x2wl', '43u8tso910dz8dt', 'xqdrviwdw', '7l7yadcte87djr', 'q15mt30n', 'qqkr1e', 'crikkya', '8wd6jen', 'faom1oe50', 'tv0v155s', 'aape6k4ezbfrs', 'es8g1pv68u7u', 'x5ul55', 'bd9ruvnmyztx', 'hoxbua37ktf', '7zogvky8fwh4jg0', 'riyn3u11', 'ak563r', 'lcijo', '64bx7o9vuuo', 'da9wiu2m', 'gh6n7rug9i0', 'jhfcl3okrtbc', 'ge6ojb', 'kvonxeaxr', 'ahsdbi0', 'l650b9ek5', 'rjmov34', 'wyk2qot8t', 'f31ys', 'tw69kk86y4vg', 'v35azec', '09ym0npvx1kda', 'l6dgbemnb1', 's3y7lzc', 'wwmkn0v9z8', 'ifj0n1wc', 'fmdl37pc', 'cwpr3r6neytrhp', '1oxk7k4w3ximbab', 'nou2dvk07x3mp', 'bpl67u', 'm8c7djwpbgb', '3y0og32d8vs', 'j9wvx9rmskz974', 'n7t25ebxl', 'sy44mmp8frwcoa', 'yo53zhl', 'k11b720wazfv', 'pyu0627irci', 'm6i47db4zpe', 'q2cexmfasyj', 'c72qkvx2e2', 'ds1ef5ird83', 'j579cj', 'ylrm918qmw', 'krnsn9hwv6', 'sk0bqobm', 'z9i4y4', 'anwr77mdpgm', 'xyba8ffmn5v7', 'qrk2aq', 'q6q6armjb', 'kxye713', 'pns4l62', 'lhoq4zpqhro', 'n5rfqbwu5', 'h02ftmqj36hu', 'w015yjn', 'm683v0ao2', 'e12nopfmvc', 'a32nf', 'et74q6tf452l2', 'rfhauosjhda6ykr', 'ou7zo86', 'kmmap7m', 'dwr25sjecd0g', 'gd01fsomej', 'y6zrw3g3kl1wm7', 'xm414rh9b', '4k8nh71q2fbh']

username_queries_with_gt = [(q, True) if q in usernames else (q, False) for q in username_queries]

def unit_test_linear_search():
    print("=== Unit test for linear search, enquiry {len(username_queries_with_gt)} times ===")
    error_count = 0
    for q, gt in username_queries_with_gt:
        occupied = check_username_linear(q, usernames)
        print(f'    {q} is occupied = {occupied}, expected {gt}')
        if occupied != gt:
            error_count += 1
    print(f'Total error count: {error_count}, accuracy: {100 - error_count / len(username_queries_with_gt) * 100}%')

def unit_test_binary_search():
    print("=== Unit test for binary search, enquiry {len(username_queries_with_gt)} times ===")
    error_count = 0
    sorted_usernames = sort_usernames(usernames)
    for q, gt in username_queries_with_gt:
        occupied = check_username_binary(q, sorted_usernames)
        print(f'    {q} is occupied = {occupied}, expected {gt}')
        if occupied != gt:
            error_count += 1
    print(f'Total error count: {error_count}, accuracy: {100 - error_count / len(username_queries_with_gt) * 100}%')

def unit_test_hash():
    print("=== Unit test for hash, enquiry {len(username_queries_with_gt)} times ===")
    username_hash_set = create_username_hash_table(usernames)
    error_count = 0
    for q, gt in username_queries_with_gt:
        occupied = check_username_hash(q, username_hash_set)
        print(f'    {q} is occupied = {occupied}, expected {gt}')
        if occupied != gt:
            error_count += 1
    print(f'Total error count: {error_count}, accuracy: {100 - error_count / len(username_queries_with_gt) * 100}%')


def unit_test_bloom_filter():
    print("=== Unit test for bloom filter, enquiry {len(username_queries_with_gt)} times ===")
    bf = BloomFilter(n=1000, p=0.01)
    for w in usernames:
        bf.add(w)
    error_count = 0
    for (q, gt) in username_queries_with_gt:
        occupied = bf.contains(q)
        print(f'    {q} is occupied = {occupied}, expected {gt}')
        if occupied != gt:
            error_count += 1
    print(f'Total error count: {error_count}, accuracy: {100 - error_count / len(username_queries_with_gt) * 100}%')

def unit_test_cuckoo_filter():
    print("=== Unit test for cuckoo filter, enquiry {len(username_queries_with_gt)} times ===")
    cf = CuckooFilter(capacity=100, bucket_size=4, fingerprint_size=8)
    for w in usernames:
        cf.add(w)

    error_count = 0
    for (q, gt) in username_queries_with_gt:
        occupied = cf.contains(q)
        print(f'    {q} is occupied in cuckoo filter {occupied}, expected {gt}')
        if occupied != gt:
            error_count += 1
    print(f'Total error count: {error_count}, accuracy: {100 - error_count / len(username_queries_with_gt) * 100}%')

    # randomly select 10 elements from usernames
    print(f'randomly selecting 10 elements from usernames and deleting them from cuckoo filter ===')
    random_elements = random.sample(usernames, 10)
    for q in random_elements:
        print(f'deleting {q} from cuckoo filter')
        cf.delete(q)
    error_count = 0 
    for q in random_elements:
        occupied = cf.contains(q)
        print(f'    {q} is occupied = {occupied}, expected "False"')
        if occupied != False:
            error_count += 1
    print(f'Total error count: {error_count}, accuracy: {100 - error_count / len(random_elements) * 100}%')


if __name__ == "__main__":
    unit_test_linear_search()
    unit_test_binary_search()
    unit_test_hash()
    unit_test_bloom_filter()
    unit_test_cuckoo_filter()