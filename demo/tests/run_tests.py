#!/usr/bin/env python
# run_tests.py - Place in project root (same level as eispy2d/ folder)

"""
Test runner for eispy2d library.
Run this script from the project root directory.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Verbose output
    python run_tests.py test_configuration.py  # Run specific test file
"""

import sys
import os
import unittest
import argparse


def get_tests_directory():
    """Find the tests directory location."""
    # Check if we're in the project root with tests subdirectory
    possible_paths = [
        'tests',                           # Running from project root
        './tests',                         # Explicit relative path
        './demo/tests',                    # Running from eispy2d/demo/scripts
        '../demo/tests',                   # Running from eispy2d/scripts
        '../../demo/tests',                # Running from deeper
        os.path.join(os.path.dirname(__file__), 'tests'),  # Relative to this file
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return os.path.abspath(path)
    
    return None


def add_project_to_path():
    """Add the project root to sys.path."""
    # Find project root (contains eispy2d folder)
    possible_roots = [
        os.getcwd(),                       # Current directory
        os.path.dirname(os.getcwd()),      # Parent directory
        os.path.join(os.getcwd(), '..'),   # One level up
        os.path.join(os.getcwd(), '../..'), # Two levels up
        os.path.join(os.path.dirname(__file__), '..'),  # Relative to this file
        os.path.join(os.path.dirname(__file__), '../..'),  # Two levels up from file
    ]
    
    for root in possible_roots:
        root = os.path.abspath(root)
        if os.path.exists(os.path.join(root, 'eispy2d')):
            if root not in sys.path:
                sys.path.insert(0, root)
                print(f"Added project root to path: {root}")
            return root
    
    # If we can't find eispy2d, try to find it by searching
    for root in possible_roots:
        root = os.path.abspath(root)
        for item in os.listdir(root):
            if item.startswith('eispy2d') and os.path.isdir(os.path.join(root, item)):
                if root not in sys.path:
                    sys.path.insert(0, root)
                    print(f"Added project root to path: {root}")
                return root
    
    return None


def run_all_tests(verbosity=1):
    """Run all test files in the tests directory."""
    tests_dir = get_tests_directory()
    if tests_dir is None:
        print("Error: Could not find 'tests' directory")
        return False
    
    print(f"Found tests directory: {tests_dir}")
    
    # Change to the tests directory parent to make discovery work
    original_dir = os.getcwd()
    os.chdir(os.path.dirname(tests_dir))
    
    try:
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover('tests', pattern='test_*.py')
        
        test_runner = unittest.TextTestRunner(verbosity=verbosity)
        result = test_runner.run(test_suite)
        
        return result.wasSuccessful()
    finally:
        os.chdir(original_dir)


def run_specific_test(test_file, verbosity=1):
    """Run a specific test file."""
    if not test_file.endswith('.py'):
        test_file += '.py'
    
    tests_dir = get_tests_directory()
    if tests_dir is None:
        print("Error: Could not find 'tests' directory")
        return False
    
    test_path = os.path.join(tests_dir, test_file)
    if not os.path.exists(test_path):
        print(f"Error: Test file '{test_path}' not found")
        return False
    
    # Change to the tests directory parent to make discovery work
    original_dir = os.getcwd()
    os.chdir(os.path.dirname(tests_dir))
    
    try:
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover('tests', pattern=test_file)
        
        test_runner = unittest.TextTestRunner(verbosity=verbosity)
        result = test_runner.run(test_suite)
        
        return result.wasSuccessful()
    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(description='Run eispy2d tests')
    parser.add_argument(
        'test_file', 
        nargs='?', 
        help='Specific test file to run (e.g., test_configuration.py)'
    )
    parser.add_argument(
        '-v', '--verbose', 
        action='store_true', 
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    verbosity = 2 if args.verbose else 1
    
    print("=" * 60)
    print("EISPY2D Test Suite")
    print("=" * 60)
    
    # Add project to path
    project_root = add_project_to_path()
    if project_root is None:
        print("Error: Could not find 'eispy2d' directory. Make sure you're in the correct location.")
        sys.exit(1)
    print(f"Project root: {project_root}")
    print()
    
    if args.test_file:
        print(f"Running: {args.test_file}")
        success = run_specific_test(args.test_file, verbosity)
    else:
        print("Running all tests...")
        success = run_all_tests(verbosity)
    
    print("\n" + "=" * 60)
    if success:
        print("All tests passed! ✓")
    else:
        print("Some tests failed! ✗")
    print("=" * 60)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()