#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Validation Script
==========================
Validate pipeline configuration and setup

Usage:
    python validate_pipeline.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def check_file_structure():
    """Check if project file structure is complete"""
    print("=" * 70)
    print("Checking project file structure...")
    print("=" * 70)
    
    required_files = {
        'Master controller': 'run_master_pipeline.py',
        'Configuration': 'config.yaml',
        'Dependencies': 'requirements.txt',
        'README': 'README.md',
    }
    
    required_dirs = {
        'Source directory': 'src',
        'Data directory': 'data',
        'Output directory': 'outputs',
        'Stage 1': 'src/stage1_data_quality',
        'Stage 2': 'src/stage2_feature_engineering',
        'Stage 3': 'src/stage3_modeling_action',
        'Stage 4': 'src/stage4_reporting',
    }
    
    # Check files
    print("\nFile Check:")
    all_files_ok = True
    for name, filepath in required_files.items():
        path = PROJECT_ROOT / filepath
        status = "[OK]" if path.exists() else "[MISSING]"
        print(f"  {status} {name}: {filepath}")
        if not path.exists():
            all_files_ok = False
    
    # Check directories
    print("\nDirectory Check:")
    all_dirs_ok = True
    for name, dirpath in required_dirs.items():
        path = PROJECT_ROOT / dirpath
        status = "[OK]" if path.exists() else "[MISSING]"
        print(f"  {status} {name}: {dirpath}")
        if not path.exists():
            all_dirs_ok = False
    
    return all_files_ok and all_dirs_ok


def check_stage_modules():
    """Check if stage modules exist"""
    print("\n" + "=" * 70)
    print("Checking stage modules...")
    print("=" * 70)
    
    stages = {
        'Stage 1': 'src/stage1_data_quality/run_data_quality.py',
        'Stage 2': 'src/stage2_feature_engineering/run_feature_strategy.py',
        'Stage 3': 'src/stage3_modeling_action/run_modeling_action.py',
        'Stage 4': 'src/stage4_reporting/build_integrated_report.py',
    }
    
    all_ok = True
    for stage_name, module_path in stages.items():
        path = PROJECT_ROOT / module_path
        status = "[OK]" if path.exists() else "[MISSING]"
        print(f"  {status} {stage_name}: {module_path}")
        if not path.exists():
            all_ok = False
            print(f"      Warning: {stage_name} main module not found!")
    
    return all_ok


def check_output_directories():
    """Check output directory structure"""
    print("\n" + "=" * 70)
    print("Checking output directory structure...")
    print("=" * 70)
    
    output_dirs = [
        'outputs/stage1_quality',
        'outputs/stage2_features',
        'outputs/stage3_modeling',
        'outputs/stage3_modeling/models',
        'outputs/reports',
    ]
    
    all_ok = True
    for dir_path in output_dirs:
        path = PROJECT_ROOT / dir_path
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"  [CREATED] {dir_path}")
        else:
            print(f"  [OK] {dir_path}")
    
    return all_ok


def check_data_files():
    """Check data files"""
    print("\n" + "=" * 70)
    print("Checking data files...")
    print("=" * 70)
    
    raw_data = PROJECT_ROOT / 'data' / 'raw' / 'data.csv'
    
    if raw_data.exists():
        print(f"  [OK] Raw data file exists: {raw_data}")
        # Check file size
        size_mb = raw_data.stat().st_size / (1024 * 1024)
        print(f"    File size: {size_mb:.2f} MB")
        return True
    else:
        print(f"  [MISSING] Raw data file not found: {raw_data}")
        print(f"    Warning: Please place raw data file at: data/raw/data.csv")
        return False


def check_python_imports():
    """Check if required Python packages are installed"""
    print("\n" + "=" * 70)
    print("Checking Python dependencies...")
    print("=" * 70)
    
    required_packages = [
        'yaml',
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'matplotlib',
        'seaborn',
        'plotly',
        'jinja2',
    ]
    
    all_ok = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            print(f"  [MISSING] {package}")
            all_ok = False
    
    if not all_ok:
        print("\n  Warning: Some dependencies not installed. Run:")
        print("     pip install -r requirements.txt")
    
    return all_ok


def create_logs_directory():
    """Create logs directory"""
    logs_dir = PROJECT_ROOT / 'logs'
    if not logs_dir.exists():
        logs_dir.mkdir(exist_ok=True)
        print(f"\n[CREATED] Logs directory: logs/")
    return True


def main():
    """Main validation process"""
    print("\n")
    print("=" * 70)
    print("          Pipeline Validation Script")
    print("=" * 70)
    print("\n")
    
    checks = [
        ("File Structure", check_file_structure),
        ("Stage Modules", check_stage_modules),
        ("Output Directories", check_output_directories),
        ("Data Files", check_data_files),
        ("Python Dependencies", check_python_imports),
        ("Logs Directory", create_logs_directory),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"\n  [FAILED] {check_name} check failed: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)
    
    for check_name, passed in results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"  {status} {check_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("All checks passed. Ready to run pipeline:")
        print("   python run_master_pipeline.py")
    else:
        print("Some checks failed. Please address the issues above.")
    print("=" * 70)
    print("\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
