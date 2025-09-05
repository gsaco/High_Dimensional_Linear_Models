#!/usr/bin/env python3
"""
Create visualization showing the overfitting phenomenon.
This script generates and saves plots demonstrating the key results.
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

def create_overfitting_plots():
    """Create plots showing the overfitting phenomenon."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate data
    n = 1000
    noise_std = 0.5
    X = np.random.uniform(-0.5, 0.5, n)
    f_X = np.exp(4 * X) - 1
    epsilon = np.random.normal(0, noise_std, n)
    Y = f_X + epsilon
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42
    )
    
    # Feature counts to test
    feature_counts = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200]
    
    results = {
        'n_features': [],
        'r2_train': [],
        'adj_r2': [],
        'r2_test': []
    }
    
    for n_features in feature_counts:
        if n_features >= len(X_train) // 3:  # Avoid overfitting issues
            continue
            
        try:
            # Create polynomial features
            poly = PolynomialFeatures(degree=n_features, include_bias=False)
            X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
            X_test_poly = poly.transform(X_test.reshape(-1, 1))
            
            # Fit model
            model = LinearRegression(fit_intercept=False)
            model.fit(X_train_poly, y_train)
            
            # Calculate metrics
            r2_train = r2_score(y_train, model.predict(X_train_poly))
            r2_test = r2_score(y_test, model.predict(X_test_poly))
            adj_r2 = calculate_adjusted_r2(r2_train, len(y_train), X_train_poly.shape[1])
            
            results['n_features'].append(n_features)
            results['r2_train'].append(r2_train)
            results['adj_r2'].append(adj_r2)
            results['r2_test'].append(r2_test)
            
        except:
            continue
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: All metrics together
    plt.subplot(1, 3, 1)
    plt.plot(results['n_features'], results['r2_train'], 'b-o', label='Training R²', markersize=4)
    plt.plot(results['n_features'], results['r2_test'], 'r-o', label='Test R²', markersize=4)
    valid_adj = [(x, y) for x, y in zip(results['n_features'], results['adj_r2']) if not np.isnan(y)]
    if valid_adj:
        x_adj, y_adj = zip(*valid_adj)
        plt.plot(x_adj, y_adj, 'g-o', label='Adjusted R²', markersize=4)
    plt.xlabel('Number of Features')
    plt.ylabel('R-squared')
    plt.title('Overfitting Demonstration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Plot 2: Focus on out-of-sample performance
    plt.subplot(1, 3, 2)
    plt.plot(results['n_features'], results['r2_test'], 'r-o', linewidth=2, markersize=6)
    plt.xlabel('Number of Features')
    plt.ylabel('Out-of-sample R²')
    plt.title('Test Performance vs Model Complexity')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Mark the optimal point
    best_idx = np.argmax(results['r2_test'])
    best_features = results['n_features'][best_idx]
    best_r2 = results['r2_test'][best_idx]
    plt.plot(best_features, best_r2, 'go', markersize=10, label=f'Optimal: {best_features} features')
    plt.legend()
    
    # Plot 3: Overfitting measure
    plt.subplot(1, 3, 3)
    overfitting = np.array(results['r2_train']) - np.array(results['r2_test'])
    plt.plot(results['n_features'], overfitting, 'purple', marker='o', linewidth=2, markersize=6)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Number of Features')
    plt.ylabel('Overfitting (Train R² - Test R²)')
    plt.title('Overfitting vs Model Complexity')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('overfitting_demonstration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results

if __name__ == "__main__":
    print("Creating overfitting demonstration plots...")
    results = create_overfitting_plots()
    print("✅ Plots created and saved as 'overfitting_demonstration.png'")
    print(f"📊 Tested {len(results['n_features'])} different model complexities")
    print(f"🎯 Optimal complexity: {results['n_features'][np.argmax(results['r2_test'])]} features")