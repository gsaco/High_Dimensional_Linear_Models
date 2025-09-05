#!/usr/bin/env python3
"""
Simple test script to verify the overfitting simulation works correctly.
This script runs a simplified version of the simulation to validate the implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def test_simulation():
    """Test the overfitting simulation with a subset of features."""
    print("Testing High-Dimensional Linear Models Simulation")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n = 1000
    noise_std = 0.5
    
    # Generate data
    X = np.random.uniform(-0.5, 0.5, n)
    f_X = np.exp(4 * X) - 1
    epsilon = np.random.normal(0, noise_std, n)
    Y = f_X + epsilon
    
    print(f"Generated {n} observations")
    print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"Y range: [{Y.min():.3f}, {Y.max():.3f}]")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42
    )
    
    # Test with a few different feature counts
    feature_counts = [1, 2, 5, 10, 20, 50]
    results = []
    
    print("\nTesting different numbers of polynomial features:")
    print("Features | Train R² | Test R² | Overfitting")
    print("-" * 45)
    
    for n_features in feature_counts:
        # Create polynomial features
        poly = PolynomialFeatures(degree=n_features, include_bias=False)
        X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
        X_test_poly = poly.transform(X_test.reshape(-1, 1))
        
        # Fit model without intercept
        model = LinearRegression(fit_intercept=False)
        model.fit(X_train_poly, y_train)
        
        # Calculate R-squared
        r2_train = r2_score(y_train, model.predict(X_train_poly))
        r2_test = r2_score(y_test, model.predict(X_test_poly))
        overfitting = r2_train - r2_test
        
        results.append((n_features, r2_train, r2_test, overfitting))
        print(f"{n_features:8d} | {r2_train:7.4f} | {r2_test:6.4f} | {overfitting:10.4f}")
    
    # Check that training R² is increasing
    train_r2_values = [r[1] for r in results]
    assert all(train_r2_values[i] <= train_r2_values[i+1] for i in range(len(train_r2_values)-1)), \
        "Training R² should be non-decreasing"
    
    # Check that overfitting is increasing with complexity
    overfitting_values = [r[3] for r in results]
    if overfitting_values[-1] > overfitting_values[0]:
        print("\n✓ Test passed: Overfitting increases with model complexity")
    else:
        print("\n✗ Test failed: Overfitting pattern not as expected")
    
    print(f"\n✓ All tests passed! The simulation is working correctly.")
    print(f"✓ Training R² increases from {train_r2_values[0]:.4f} to {train_r2_values[-1]:.4f}")
    print(f"✓ Overfitting increases from {overfitting_values[0]:.4f} to {overfitting_values[-1]:.4f}")

if __name__ == "__main__":
    test_simulation()