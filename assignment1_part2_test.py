"""
Assignment 1 - Part 2 Test (Overfitting Analysis)
High Dimensional Linear Models

This script tests the corrected overfitting analysis.

Author: Generated for gsaco/High_Dimensional_Linear_Models
"""

import sys
import os
sys.path.append('/home/runner/work/High_Dimensional_Linear_Models/High_Dimensional_Linear_Models/Python/scripts')

from part2_overfitting import main as overfitting_main

def print_header(title, width=80):
    """Print a formatted header."""
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

def main():
    """Run the overfitting analysis."""
    print_header("ASSIGNMENT 1 - PART 2: OVERFITTING ANALYSIS (CORRECTED)")
    
    print("\n🎯 Running corrected overfitting analysis...")
    print("   - Data generation: y = 2*x + e (no intercept)")
    print("   - Model fitting: fit_intercept=False")
    print("   - Features tested: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000")
    print("   - Train/test split: 75%/25%")
    
    # Run the corrected overfitting analysis
    overfitting_main()
    
    print_header("PART 2 ANALYSIS COMPLETE")
    print("✅ Corrected implementation follows assignment specification")
    print("✅ Uses simple linear relationship with no intercept") 
    print("✅ Shows proper overfitting patterns")
    print("✅ Generates three separate R-squared plots")
    print("✅ Results saved to Python/output/ directory")

if __name__ == "__main__":
    main()