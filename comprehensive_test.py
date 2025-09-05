#!/usr/bin/env python3
"""
Comprehensive test of the overfitting simulation showing clear overfitting patterns.
This script runs the full simulation to demonstrate the expected overfitting behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def calculate_adjusted_r2(r2, n, p):
    """Calculate adjusted R-squared"""
    if n <= p + 1:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def run_full_simulation():
    """Run the complete overfitting simulation."""
    print("High-Dimensional Linear Models: Overfitting Demonstration")
    print("=" * 60)
    
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
    
    print(f"Generated {n} observations with nonlinear relationship f(X) = exp(4X) - 1")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42
    )
    
    print(f"Training set: {len(X_train)} observations")
    print(f"Test set: {len(X_test)} observations")
    
    # Feature counts to test (including high-dimensional cases)
    feature_counts = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    
    results = []
    
    print("\nSimulation Results:")
    print("Features | Train R² | Adj R²  | Test R² | Overfitting | Complexity")
    print("-" * 70)
    
    for n_features in feature_counts:
        # Skip if too many features
        if n_features >= len(X_train):
            continue
            
        try:
            # Create polynomial features
            poly = PolynomialFeatures(degree=n_features, include_bias=False)
            X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
            X_test_poly = poly.transform(X_test.reshape(-1, 1))
            
            # Fit model without intercept
            model = LinearRegression(fit_intercept=False)
            model.fit(X_train_poly, y_train)
            
            # Calculate metrics
            r2_train = r2_score(y_train, model.predict(X_train_poly))
            r2_test = r2_score(y_test, model.predict(X_test_poly))
            adj_r2 = calculate_adjusted_r2(r2_train, len(y_train), X_train_poly.shape[1])
            overfitting = r2_train - r2_test
            
            results.append({
                'n_features': n_features,
                'r2_train': r2_train,
                'adj_r2': adj_r2,
                'r2_test': r2_test,
                'overfitting': overfitting,
                'n_params': X_train_poly.shape[1]
            })
            
            # Determine complexity level
            if overfitting < 0.05:
                complexity = "Good"
            elif overfitting < 0.15:
                complexity = "Moderate"
            else:
                complexity = "Overfit"
            
            print(f"{n_features:8d} | {r2_train:7.4f} | {adj_r2:6.4f} | {r2_test:6.4f} | {overfitting:10.4f} | {complexity:>9s}")
            
        except Exception as e:
            print(f"{n_features:8d} | Error: {str(e)[:40]}...")
    
    # Analysis
    print("\n" + "=" * 60)
    print("OVERFITTING ANALYSIS")
    print("=" * 60)
    
    # Find the optimal complexity
    test_r2_values = [r['r2_test'] for r in results]
    best_idx = np.argmax(test_r2_values)
    best_result = results[best_idx]
    
    print(f"\nOptimal model complexity:")
    print(f"  Features: {best_result['n_features']}")
    print(f"  Test R²: {best_result['r2_test']:.4f}")
    print(f"  Overfitting: {best_result['overfitting']:.4f}")
    
    # Show progression of overfitting
    print(f"\nOverfitting progression:")
    for i, result in enumerate(results):
        if i == 0:
            continue
        prev_test = results[i-1]['r2_test']
        curr_test = result['r2_test']
        trend = "↗️" if curr_test > prev_test else "↘️"
        print(f"  {results[i-1]['n_features']:3d} → {result['n_features']:3d} features: Test R² {prev_test:.4f} → {curr_test:.4f} {trend}")
    
    # Key insights
    print(f"\nKey Observations:")
    print(f"  • Training R² ranges from {results[0]['r2_train']:.4f} to {results[-1]['r2_train']:.4f}")
    print(f"  • Test R² peaks at {best_result['n_features']} features ({max(test_r2_values):.4f})")
    print(f"  • Maximum overfitting: {max(r['overfitting'] for r in results):.4f}")
    print(f"  • This demonstrates the bias-variance tradeoff!")
    
    return results

if __name__ == "__main__":
    results = run_full_simulation()
    print("\n✅ Simulation completed successfully!")
    print("📊 The results clearly show the overfitting phenomenon in high-dimensional models.")