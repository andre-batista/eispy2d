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
sys.path.insert(1, '../../../eispy2d/')


import unittest
import argparse



def run_all_tests(verbosity=1):
    """Run all test files in the tests directory."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    test_runner = unittest.TextTestRunner(verbosity=verbosity)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()


def run_specific_test(test_file, verbosity=1):
    """Run a specific test file."""
    if not test_file.endswith('.py'):
        test_file += '.py'
    
    test_path = os.path.join('tests', test_file)
    if not os.path.exists(test_path):
        print(f"Error: Test file '{test_path}' not found")
        return False
    
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern=test_file)
    
    test_runner = unittest.TextTestRunner(verbosity=verbosity)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()


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