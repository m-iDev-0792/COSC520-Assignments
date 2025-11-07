from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal
import math
import random
import time
import tracemalloc
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil

from dataset.dataset_load import *
from spatial_trees.kd_tree_2d import *
from spatial_trees.implicit_kd_tree_2d import *
from spatial_trees.quadtree_2d import *

def load_points_simple(directory: str) -> np.ndarray:
    try:
        from dataset.dataset_load import load_points
        return load_points(directory)
    except:
        # Generate random points as fallback
        print("Dataset not found, generating random points...")
        np.random.seed(42)
        return np.random.random((10000000, 2)).astype(np.float64)


def benchmark_all_spatial_trees():
    """
    Comprehensive benchmark comparing:
    - LinearSearch2D
    - QuadTree2D
    - KDTree2D (variance)
    - KDTree2D (interleave)
    - ImplicitKDTree2D (variance)
    - ImplicitKDTree2D (interleave)
    """
    
    # Configuration
    N_values = [10000, 50000, 100000, 200000, 500000, 1000000]
    N_values = [10000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000]
    query_count = 1000  # Fixed number of queries per test
    results = []

    # Load or generate dataset
    print("="*80)
    print("SPATIAL TREES BENCHMARK - Comprehensive Comparison")
    print("="*80)
    print(f'\nLoading points from directory ./dataset')
    
    all_points = load_points_simple("./dataset")
    points_num = len(all_points)
    print(f'Loaded/Generated {points_num} points')
    
    # Split into source and query sets
    query_num_ratio = 0.2
    source_point_num = points_num - round(points_num * query_num_ratio)
    source_point_set = all_points[:source_point_num]
    query_point_set = all_points[source_point_num:]
    
    print(f'Source points: {len(source_point_set)}, Query points: {len(query_point_set)}')
    print("="*80)

    # Create output directory
    if not os.path.exists("./output"):
        os.makedirs("./output")
    
    # Run benchmarks for each N
    for N in N_values:
        points = sample_n_elements(source_point_set, N)
        queries = sample_n_elements(query_point_set, query_count)
        
        print(f"\n{'='*80}")
        print(f"N = {N:,} points ({query_count} queries)")
        print(f"{'='*80}")
        
        # ========================================
        # 1. Linear Search
        # ========================================
        print("\n[1/6] LinearSearch2D...")
        try:
            tracemalloc.start()
            t0 = time.perf_counter()
            linear = LinearSearch2D(points)
            build_time = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            t1 = time.perf_counter()
            for q in queries:
                linear.nearest(tuple(q))
            query_time = time.perf_counter() - t1
            
            results.append({
                "N": N,
                "tree_type": "Linear",
                "method": "N/A",
                "build_time_s": build_time,
                "query_time_s": query_time,
                "avg_query_ms": query_time * 1000 / query_count,
                "peak_MB": peak / 1e6,
            })
            print(f"  ✓ Build: {build_time:.4f}s | Query: {query_time:.4f}s | Mem: {peak/1e6:.2f}MB")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        # ========================================
        # 2. QuadTree2D
        # ========================================
        print("\n[2/6] QuadTree2D...")
        try:
            tracemalloc.start()
            t0 = time.perf_counter()
            quadtree = QuadTree2D(points)
            build_time = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            t1 = time.perf_counter()
            for q in queries:
                quadtree.nearest(tuple(q))
            query_time = time.perf_counter() - t1
            
            results.append({
                "N": N,
                "tree_type": "QuadTree",
                "method": "N/A",
                "build_time_s": build_time,
                "query_time_s": query_time,
                "avg_query_ms": query_time * 1000 / query_count,
                "peak_MB": peak / 1e6,
            })
            print(f"  ✓ Build: {build_time:.4f}s | Query: {query_time:.4f}s | Mem: {peak/1e6:.2f}MB")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        
        # ========================================
        # 3-4. KDTree2D (2 variants)
        # ========================================
        config_num = 3
        for method in ["variance", "interleave"]:
            print(f"\n[{config_num}/6] KDTree2D ({method})...")
            config_num += 1
            
            try:
                tracemalloc.start()
                t0 = time.perf_counter()
                tree = KDTree2D(points, split_method=method, approx_median=False)
                build_time = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                t1 = time.perf_counter()
                for q in queries:
                    tree.nearest(tuple(q))
                query_time = time.perf_counter() - t1
                
                results.append({
                    "N": N,
                    "tree_type": "KDTree",
                    "method": method,
                    "build_time_s": build_time,
                    "query_time_s": query_time,
                    "avg_query_ms": query_time * 1000 / query_count,
                    "peak_MB": peak / 1e6,
                })
                print(f"  ✓ Build: {build_time:.4f}s | Query: {query_time:.4f}s | Mem: {peak/1e6:.2f}MB")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
        
        # ========================================
        # 5-6. ImplicitKDTree2D (2 variants)
        # ========================================
        for method in ["variance", "interleave"]:
            print(f"\n[{config_num}/6] ImplicitKDTree2D ({method})...")
            config_num += 1
            
            try:
                tracemalloc.start()
                t0 = time.perf_counter()
                tree = ImplicitKDTree2D(points, split_method=method, approx_median=False)
                build_time = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                t1 = time.perf_counter()
                for q in queries:
                    tree.nearest(tuple(q))
                query_time = time.perf_counter() - t1
                
                results.append({
                    "N": N,
                    "tree_type": "ImplicitKDTree",
                    "method": method,
                    "build_time_s": build_time,
                    "query_time_s": query_time,
                    "avg_query_ms": query_time * 1000 / query_count,
                    "peak_MB": peak / 1e6,
                })
                print(f"  ✓ Build: {build_time:.4f}s | Query: {query_time:.4f}s | Mem: {peak/1e6:.2f}MB")
            except Exception as e:
                print(f"  ✗ Failed: {e}")

    # ========================================
    # Save Results
    # ========================================
    df = pd.DataFrame(results)
    csv_path = "./output/spatial_trees_comprehensive_benchmark.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print(f"✅ Results saved to {csv_path}")
    print("="*80)
    
    # ========================================
    # Generate Plots
    # ========================================
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Define colors and styles for each tree type
    tree_styles = {
        'Linear': {'color': 'black', 'linestyle': '-', 'marker': 's', 'label': 'Linear Search'},
        'QuadTree': {'color': 'purple', 'linestyle': '-', 'marker': 'D', 'label': 'QuadTree'},
        'KDTree_variance': {'color': 'blue', 'linestyle': '-', 'marker': 'o', 'label': 'KDTree (variance)'},
        'KDTree_interleave': {'color': 'green', 'linestyle': '-', 'marker': '^', 'label': 'KDTree (interleave)'},
        'ImplicitKDTree_variance': {'color': 'red', 'linestyle': '-', 'marker': 's', 'label': 'ImplicitKD (variance)'},
        'ImplicitKDTree_interleave': {'color': 'orange', 'linestyle': '-', 'marker': 'v', 'label': 'ImplicitKD (interleave)'},
    }
    
    def get_config_key(row):
        if row['tree_type'] == 'Linear':
            return 'Linear'
        elif row['tree_type'] == 'QuadTree':
            return 'QuadTree'
        else:
            return f"{row['tree_type']}_{row['method']}"
    
    df['config'] = df.apply(get_config_key, axis=1)
    
    # ========================================
    # Plot 1: Build Time
    # ========================================
    ax = axes[0]
    for config, style in tree_styles.items():
        data = df[df['config'] == config].sort_values('N')
        if len(data) > 0:
            ax.plot(data['N'], data['build_time_s'], 
                   linewidth=2, markersize=7, **style)
    
    ax.set_xlabel("N (number of points)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Build Time (seconds)", fontsize=12, fontweight='bold')
    ax.set_title("Build Time Comparison", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.set_xscale('log')
    
    # ========================================
    # Plot 2: Query Time
    # ========================================
    ax = axes[1]
    for config, style in tree_styles.items():
        data = df[df['config'] == config].sort_values('N')
        if len(data) > 0:
            ax.plot(data['N'], data['query_time_s'],
                   linewidth=2, markersize=7, **style)
    
    ax.set_xlabel("N (number of points)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Query Time (seconds)", fontsize=12, fontweight='bold')
    ax.set_title(f"Query Time Comparison ({query_count} queries)", fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.set_xscale('log')
    
    # ========================================
    # Plot 3: Memory Usage
    # ========================================
    ax = axes[2]
    for config, style in tree_styles.items():
        data = df[df['config'] == config].sort_values('N')
        if len(data) > 0:
            ax.plot(data['N'], data['peak_MB'],
                   linewidth=2, markersize=7, **style)
    
    ax.set_xlabel("N (number of points)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Peak Memory (MB)", fontsize=12, fontweight='bold')
    ax.set_title("Memory Usage Comparison", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, loc='best', framealpha=0.9)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plot_path = "./output/spatial_trees_comprehensive_benchmark.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plots saved to {plot_path}")
    
    # ========================================
    # Generate Summary Statistics
    # ========================================
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (at largest N)")
    print("="*80)
    
    max_n = df['N'].max()
    summary = df[df['N'] == max_n][['tree_type', 'method', 
                                      'build_time_s', 'query_time_s', 'avg_query_ms', 'peak_MB']].copy()
    summary = summary.sort_values('query_time_s')
    
    print(f"\nResults for N = {max_n:,}:")
    print(summary.to_string(index=False))
    
    # Find best performers
    best_build = df[df['N'] == max_n].loc[df[df['N'] == max_n]['build_time_s'].idxmin()]
    best_query = df[df['N'] == max_n].loc[df[df['N'] == max_n]['query_time_s'].idxmin()]
    best_memory = df[df['N'] == max_n].loc[df[df['N'] == max_n]['peak_MB'].idxmin()]
    
    print("\n" + "="*80)
    print("BEST PERFORMERS:")
    print("="*80)
    print(f"Fastest Build:  {best_build['config']} ({best_build['build_time_s']:.4f}s)")
    print(f"Fastest Query:  {best_query['config']} ({best_query['avg_query_ms']:.4f}ms avg)")
    print(f"Lowest Memory:  {best_memory['config']} ({best_memory['peak_MB']:.2f}MB)")
    
    plt.show()
    
    return df


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Starting Comprehensive Spatial Trees Benchmark")
    print("="*80)
    
    df = benchmark_all_spatial_trees()
    
    print("\n" + "="*80)
    print("✅ Benchmark Complete!")
    print("="*80)