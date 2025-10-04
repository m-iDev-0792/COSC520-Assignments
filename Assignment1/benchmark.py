import os, sys
import time
from typing import List, Set, Dict
from simple_login_checker_algo import *
import os, psutil, time, threading
from pympler import asizeof
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import random
from dataset_gen import *
from bloom_filter import *
from cuckoo_filter import *
import numpy as np


def monitor_memory(interval=0.5, stop_flag=None, results=None):
    process = psutil.Process(os.getpid())
    peak = 0
    samples = []

    while not stop_flag["stop"]:
        mem = process.memory_info().rss  # Resident Set Size in bytes
        samples.append(mem)
        peak = max(peak, mem)
        time.sleep(interval)

    results["peak"] = peak
    results["avg"] = sum(samples) / len(samples)

def get_correctness(list1, list2):
    total = len(list1)
    matches = sum(v1 == v2 for (_, v1), (_, v2) in zip(list1, list2))
    return matches / total * 100

def benchmark_general_framework(query_list, login_checker, enable_memory_monitor = False) -> Dict[str, float]:
    print(f"======== Benchmarking {login_checker.__class__.__name__} with {len(query_list)} usernames ========")
    ret = {}
    # check time and memory usage
    query_size = len(query_list)
    real_query_size = query_size
    if query_size*login_checker.dataset_size > 10000*10000 and "Linear" in login_checker.__class__.__name__ :
        real_query_size = int(10000*10000/login_checker.dataset_size)
        query_list = random.sample(query_list, real_query_size) # resample the query list to 5000 when it's LinearChecker
        print(f"Resampled query {query_size} ->{real_query_size} usernames")
    
    if enable_memory_monitor: # monitor memory
        stop_flag = {"stop": False}
        mem_stat = {}

        monitor_thread = threading.Thread(
            target=monitor_memory, args=(0.5, stop_flag, mem_stat)
        )
        monitor_thread.start()

    start_time = time.time()
    checker_results = login_checker.check(query_list)
    time_taken = time.time() - start_time

    if enable_memory_monitor: # monitor memory
        stop_flag["stop"] = True
        monitor_thread.join()
        ret["memory_peak"] = mem_stat["peak"]
        ret["memory_avg"] = mem_stat["avg"]


    ret["algorithm"] = login_checker.__class__.__name__
    ret["query_size"] = query_size
    ret["dataset_size"] = login_checker.dataset_size
    ret["total_time"] = time_taken * query_size / real_query_size
    ret["average_time"] = time_taken / real_query_size
    ret["object_size"] = asizeof.asizeof(login_checker)
    ret["correctness"] = get_correctness(checker_results, query_list)
    print(f'Benchmark report:')
    for key, value in ret.items():
        print(f'\t{key}: {value}')
    return ret


def benchmark_single_batch(true_username_list, false_username_list, query_size):
    datasize = len(true_username_list)
    username_list_true = random.sample(true_username_list, int(query_size//2))
    username_list_false = random.sample(false_username_list, query_size - int(query_size//2))
    username_list_with_gt = [(q, True) for q in username_list_true] + [(q, False) for q in username_list_false]
    
    checkers = []
    linear_checker = LinearChecker(true_username_list)
    binary_checker = BinaryChecker(true_username_list)
    hash_checker = HashChecker(true_username_list)
    bloom_filter = BloomFilter(datasize, 0.01, true_username_list)
    cuckoo_params = cal_cuckoo_params(true_username_list, 0.01)
    cuckoo_filter = CuckooFilter(**cuckoo_params)
    checkers.append(linear_checker)
    checkers.append(binary_checker)
    checkers.append(hash_checker)
    checkers.append(bloom_filter)
    checkers.append(cuckoo_filter)

    results = []
    for checker in checkers:
        ret = benchmark_general_framework(username_list_with_gt, checker)
        results.append(ret)
    return results
    

def benchmark_10percent(test_datasize_list):
    max_size = max(test_datasize_list)
    true_username_pool = load_user_ids_from_dir("dataset", max_size)
    false_username_pool = load_user_ids_from_dir("dataset_false", max_size)
    results = []
    for datasize in test_datasize_list:
        true_username_list = random.sample(true_username_pool, datasize)
        false_username_list = random.sample(false_username_pool, datasize)
        results.extend(benchmark_single_batch(true_username_list, false_username_list, int(datasize * 0.1)))
    return results

def benchmark_10K(test_datasize_list):
    max_size = max(test_datasize_list)
    true_username_pool = load_user_ids_from_dir("dataset", max_size)
    false_username_pool = load_user_ids_from_dir("dataset_false", max_size)
    results = []
    for datasize in test_datasize_list:
        true_username_list = random.sample(true_username_pool, datasize)
        false_username_list = random.sample(false_username_pool, datasize)
        results.extend(benchmark_single_batch(true_username_list, false_username_list, 1000))
    return results

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_experiment_results(
    data_list,
    jitter_threshold=0.05,
    jitter_ratio=0.02,
    overall_jitter=True
):
    """
    绘制实验结果 (1x4 子图布局)，自动添加 jitter 避免曲线重叠，带单位
    :param data_list: 输入数据
    :param jitter_threshold: 曲线相似度阈值
    :param jitter_ratio: 抖动幅度占全局 y 值范围的比例
    :param overall_jitter: True=整条曲线整体偏移, False=每个点单独 jitter
    """
    # 指标和单位
    metrics = ["total_time", "average_time", "object_size", "correctness"]
    metric_units = {
        "total_time": "Seconds (s)",
        "average_time": "Seconds (s)",
        "object_size": "Bytes (B)",
        "correctness": "%"
    }
    
    results = defaultdict(lambda: defaultdict(list))
    for entry in data_list:
        algo = entry["algorithm"]
        ds = entry["dataset_size"]
        for metric in metrics:
            results[algo][metric].append((ds, entry[metric]))
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        curves = {}
        
        # 收集数据
        for algo, algo_data in results.items():
            data_sorted = sorted(algo_data[metric], key=lambda x: x[0])
            x = np.array([ds for ds, _ in data_sorted])
            y = np.array([val for _, val in data_sorted])
            curves[algo] = (x, y)
        
        # 计算全局 y 值范围（基于所有曲线）
        all_y_values = np.concatenate([y for _, y in curves.values()])
        global_y_min = np.min(all_y_values)
        global_y_max = np.max(all_y_values)
        global_y_range = global_y_max - global_y_min
        
        # 如果所有值都相同，使用平均值作为范围参考
        if global_y_range == 0:
            global_y_range = np.mean(all_y_values) if np.mean(all_y_values) != 0 else 1.0
        
        # 基于全局范围计算 jitter 幅度
        jitter_amount = jitter_ratio * global_y_range
        
        algos = list(curves.keys())
        adjusted_curves = {}
        overlap_groups = []  # 存储重叠的曲线组
        
        # 检测重叠曲线并分组
        for idx, algo in enumerate(algos):
            x, y = curves[algo]
            found_group = False
            
            # 检查是否与已有组重叠
            for group in overlap_groups:
                for other_algo in group:
                    _, y_other = curves[other_algo]
                    if len(y) == len(y_other):
                        # 使用全局范围归一化差异
                        diff = np.mean(np.abs(y - y_other)) / (global_y_range + 1e-9)
                        if diff < jitter_threshold:
                            group.append(algo)
                            found_group = True
                            break
                if found_group:
                    break
            
            # 如果没有找到重叠组，创建新组
            if not found_group:
                overlap_groups.append([algo])
        
        # 为每条曲线应用 jitter
        for group in overlap_groups:
            if len(group) > 1:
                # 有重叠，分配不同的偏移
                for offset_idx, algo in enumerate(group):
                    x, y = curves[algo]
                    
                    if overall_jitter:
                        # 整体偏移：在组内均匀分布
                        shift = jitter_amount * (2 * offset_idx / (len(group) - 1) - 1)
                        y = y + shift
                    else:
                        # 每点偏移：添加小范围随机抖动
                        jitter = np.random.uniform(
                            -jitter_amount * 0.5, 
                            jitter_amount * 0.5, 
                            size=len(y)
                        )
                        # 加上组内的整体偏移
                        shift = jitter_amount * (2 * offset_idx / (len(group) - 1) - 1)
                        y = y + shift + jitter
                    
                    adjusted_curves[algo] = (x, y)
            else:
                # 无重叠，保持原样
                adjusted_curves[group[0]] = curves[group[0]]
        
        # 绘制所有曲线
        for algo, (x, y) in adjusted_curves.items():
            ax.plot(x, y, marker="o", label=algo, linewidth=2, markersize=6)
        
        ax.set_xlabel("Dataset Size", fontsize=11)
        ax.set_ylabel(f"{metric} [{metric_units.get(metric, '')}]", fontsize=11)
        ax.set_title(f"{metric} vs Dataset Size", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    test_datasize_list = [
                            10000, 40000, 80000,
                            100000, 400000, 800000
                            ]
    # test_datasize_list = [10000, 50000]
    results_10K = benchmark_10K(test_datasize_list)
    with open("results_10K.json", "w") as f:
        json.dump(results_10K, f)
    

    results_10percent = benchmark_10percent(test_datasize_list)
    with open("results_10percent.json", "w") as f:
        json.dump(results_10percent, f)
    plot_experiment_results(results_10K)
    plot_experiment_results(results_10percent)
    

    