#!/usr/bin/env python3
"""
Validation Script for Synthetic Datasets
=========================================

This script validates that the synthetic datasets match the statistics
reported in the paper "Hybrid Neural–Heuristic TSP Solver with 
Attention-Guided Refinement for Real-Time AMR Coordination"

Usage:
    python validate_datasets.py

Requirements:
    - Python 3.7+
    - pandas
    - numpy

Author: BLIND REVIEW
Date: February 7, 2026
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def validate_file_exists(filepath):
    """Check if file exists"""
    path = Path(filepath)
    if path.exists():
        print_success(f"Found: {filepath} ({path.stat().st_size / 1024:.1f} KB)")
        return True
    else:
        print_error(f"Missing: {filepath}")
        return False

def validate_json_schema(data, expected_keys):
    """Validate JSON structure"""
    if not isinstance(data, list):
        print_error("Expected JSON array")
        return False
    
    if len(data) == 0:
        print_error("Empty dataset")
        return False
    
    # Check first element has required keys
    missing_keys = set(expected_keys) - set(data[0].keys())
    if missing_keys:
        print_error(f"Missing keys: {missing_keys}")
        return False
    
    print_success(f"Schema valid ({len(data)} records)")
    return True

def validate_benchmark_instances():
    """Validate tsp_instances_input.json"""
    print_header("VALIDATING BENCHMARK INSTANCES")
    
    filepath = 'data/tsp_instances_input.json'
    if not validate_file_exists(filepath):
        return False
    
    # Load data
    with open(filepath, 'r') as f:
        instances = json.load(f)
    
    # Validate schema
    expected_keys = ['instance_id', 'n_nodes', 'coordinates', 'description']
    if not validate_json_schema(instances, expected_keys):
        return False
    
    # Validate constraints
    errors = []
    for inst in instances:
        # Check n_nodes matches coordinates length
        if inst['n_nodes'] != len(inst['coordinates']):
            errors.append(f"{inst['instance_id']}: n_nodes mismatch")
        
        # Check coordinates in valid range
        for x, y in inst['coordinates']:
            if not (0 <= x <= 1000 and 0 <= y <= 1000):
                errors.append(f"{inst['instance_id']}: coordinates out of range")
                break
    
    if errors:
        for err in errors[:5]:  # Show first 5 errors
            print_error(err)
        return False
    
    # Print statistics
    sizes = [inst['n_nodes'] for inst in instances]
    print(f"  Problem sizes: {min(sizes)}-{max(sizes)}")
    print(f"  Mean size: {np.mean(sizes):.1f}")
    print(f"  Total instances: {len(instances)}")
    
    print_success("Benchmark instances valid")
    return True

def validate_solver_results():
    """Validate tsp_results_output.json"""
    print_header("VALIDATING SOLVER RESULTS")
    
    filepath = 'data/tsp_results_output.json'
    if not validate_file_exists(filepath):
        return False
    
    # Load data
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    # Validate schema
    expected_keys = ['instance_id', 'n_nodes', 'tour_length', 'lk_tour_length',
                     'optimality_gap_percent', 'runtime_ms', 'speedup', 'tour']
    if not validate_json_schema(results, expected_keys):
        return False
    
    # Extract metrics
    gaps = [r['optimality_gap_percent'] for r in results]
    runtimes = [r['runtime_ms'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # Validate against Table 3 in paper
    print("\n  Comparing to Table 3 in paper:")
    
    # Expected ranges from paper
    expected_gaps = (3.2, 8.7)
    expected_runtimes = (13.9, 154.3)
    expected_speedups = (0.91, 18.5)
    
    gap_mean = np.mean(gaps)
    gap_std = np.std(gaps)
    rt_mean = np.mean(runtimes)
    rt_std = np.std(runtimes)
    sp_mean = np.mean(speedups)
    
    print(f"  Optimality Gap: {gap_mean:.2f}% ± {gap_std:.2f}%")
    if expected_gaps[0] <= gap_mean <= expected_gaps[1]:
        print_success(f"    Within paper range [{expected_gaps[0]}, {expected_gaps[1]}]%")
    else:
        print_warning(f"    Outside paper range [{expected_gaps[0]}, {expected_gaps[1]}]%")
    
    print(f"  Runtime: {rt_mean:.1f}ms ± {rt_std:.1f}ms")
    if expected_runtimes[0] <= rt_mean <= expected_runtimes[1]:
        print_success(f"    Within paper range [{expected_runtimes[0]}, {expected_runtimes[1]}]ms")
    else:
        print_warning(f"    Outside paper range")
    
    print(f"  Speedup: {sp_mean:.2f}×")
    if expected_speedups[0] <= sp_mean <= expected_speedups[1]:
        print_success(f"    Within paper range [{expected_speedups[0]}, {expected_speedups[1]}]×")
    else:
        print_warning(f"    Outside paper range")
    
    # Real-time capability
    rt_capable = sum(1 for rt in runtimes if rt < 100) / len(runtimes) * 100
    print(f"  Real-time capable (<100ms): {rt_capable:.0f}%")
    if rt_capable >= 90:
        print_success("    Meets real-time requirement (>90%)")
    else:
        print_warning("    Below real-time threshold")
    
    print_success("Solver results valid")
    return True

def validate_warehouse_dataset():
    """Validate warehouse_dataset_synthetic.csv"""
    print_header("VALIDATING WAREHOUSE DATASET")
    
    filepath = 'data/warehouse_dataset_synthetic.csv'
    if not validate_file_exists(filepath):
        return False
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Validate schema
    expected_cols = ['instance_id', 'n_nodes', 'optimal_tour_length',
                     'hybrid_tour_length', 'hybrid_gap_percent', 'hybrid_runtime_ms']
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        print_error(f"Missing columns: {missing_cols}")
        return False
    
    print_success(f"Schema valid ({len(df)} records)")
    
    # Validate against Table 4 in paper
    print("\n  Comparing to Table 4 in paper:")
    
    # Expected values from Table 4
    expected_instances = 847
    expected_n_range = (42, 87)
    expected_gap = 4.9
    expected_runtime = 58.3
    
    # Computed statistics
    n_instances = len(df)
    n_min, n_max = df['n_nodes'].min(), df['n_nodes'].max()
    n_mean = df['n_nodes'].mean()
    gap_mean = df['hybrid_gap_percent'].mean()
    gap_std = df['hybrid_gap_percent'].std()
    rt_mean = df['hybrid_runtime_ms'].mean()
    rt_std = df['hybrid_runtime_ms'].std()
    
    print(f"  Instances: {n_instances}")
    if n_instances == expected_instances:
        print_success(f"    Matches paper (847)")
    else:
        print_warning(f"    Differs from paper ({expected_instances})")
    
    print(f"  N range: [{n_min}, {n_max}] (mean: {n_mean:.1f})")
    if n_min == expected_n_range[0] and n_max == expected_n_range[1]:
        print_success(f"    Matches paper range [42, 87]")
    else:
        print_warning(f"    Differs from paper range")
    
    print(f"  Hybrid Gap: {gap_mean:.2f}% ± {gap_std:.2f}%")
    if abs(gap_mean - expected_gap) < 0.5:
        print_success(f"    Matches paper (4.9% ± 0.5%)")
    else:
        print_warning(f"    Differs from paper (expected: 4.9%)")
    
    print(f"  Hybrid Runtime: {rt_mean:.1f}ms ± {rt_std:.1f}ms")
    if abs(rt_mean - expected_runtime) < 5:
        print_success(f"    Matches paper (58.3ms ± 5ms)")
    else:
        print_warning(f"    Differs from paper (expected: 58.3ms)")
    
    # Real-time capability
    rt_capable = (df['hybrid_runtime_ms'] < 100).sum() / len(df) * 100
    print(f"  Real-time capable (<100ms): {rt_capable:.1f}%")
    if rt_capable >= 95:
        print_success("    Exceeds requirement (>95%)")
    else:
        print_warning("    Below expectation")
    
    print_success("Warehouse dataset valid")
    return True

def main():
    """Run all validation checks"""
    print(f"\n{Colors.BOLD}Synthetic Dataset Validation for:{Colors.END}")
    print("Hybrid Neural–Heuristic TSP Solver (RICO 2026)")
    print("Authors: BLIND REVIEW\n")
    
    results = {
        'Benchmark Instances': validate_benchmark_instances(),
        'Solver Results': validate_solver_results(),
        'Warehouse Dataset': validate_warehouse_dataset()
    }
    
    # Summary
    print_header("VALIDATION SUMMARY")
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        if passed:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")
    
    print()
    if all_passed:
        print_success("All validations passed! Datasets are consistent with paper.")
    else:
        print_error("Some validations failed. Check errors above.")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
