"""
Part 2: Overfitting Analysis
Module containing functions for overfitting analysis with corrected data generation process.

This module implements the overfitting analysis following the assignment specification:
y = 2*x + e (no intercept, simple linear relationship)

Author: Generated for gsaco/High_Dimensional_Linear_Models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
import os

warnings.filterwarnings('ignore')


def generate_data(n=1000, seed=42):
    """
    Generate data following the assignment specification:
    y = 2*x + e (no intercept, simple linear relationship)
    
    Parameters:
    -----------
    n : int
        Sample size (default: 1000)
    seed : int
        Random seed for reproducibility (42)
        
    Returns:
    --------
    x : numpy.ndarray
        Feature matrix (n x 1) - sorted uniform random variables  
    y : numpy.ndarray
        Target variable (n,) following y = 2*x + e (no intercept)
    """
    np.random.seed(seed)
    
    # Generate x from uniform distribution and sort
    x = np.random.uniform(0, 1, n)
    x.sort()
    x = x.reshape(-1, 1)
    
    # Generate error term
    e = np.random.normal(0, 1, n)
    
    # Generate y with simple linear relationship (no intercept): y = 2*x + e
    y = 2.0 * x.ravel() + e
    
    return x, y


def create_polynomial_features(x, n_features):
    """
    Create polynomial features up to n_features.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Original feature matrix (n x 1)
    n_features : int
        Number of features to create
        
    Returns:
    --------
    x_poly : numpy.ndarray
        Extended feature matrix with polynomial features
    """
    n_samples = x.shape[0]
    x_poly = np.zeros((n_samples, n_features))
    
    for i in range(n_features):
        x_poly[:, i] = x.ravel() ** (i + 1)  # x^1, x^2, x^3, etc.
    
    return x_poly


def calculate_adjusted_r2(r2, n, k):
    """
    Calculate adjusted R-squared.
    
    Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - k - 1)]
    
    Parameters:
    -----------
    r2 : float
        R-squared value
    n : int
        Sample size
    k : int
        Number of features (excluding intercept)
        
    Returns:
    --------
    adj_r2 : float
        Adjusted R-squared
    """
    # Handle edge cases where we have too many features
    if n - k - 1 <= 0:
        return np.nan
    
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    return adj_r2


def overfitting_analysis():
    """
    Main function to perform overfitting analysis.
    
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame containing results for different numbers of features
    """
    print("Generating data following assignment specification: y = 2*x + e (no intercept)")
    
    # Generate the data following assignment specification
    x, y = generate_data(n=1000, seed=42)
    
    print(f"Generated data with n={len(y)} observations")
    print(f"True relationship: y = 2*x + e (no intercept)")
    print(f"x range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"y range: [{y.min():.4f}, {y.max():.4f}]")
    
    # Number of features to test (as specified)
    n_features_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    # Storage for results
    results = []
    
    print("\nAnalyzing overfitting for different numbers of features...")
    print("Features | R² (full) | Adj R² (full) | R² (out-of-sample)")
    print("-" * 60)
    
    for n_feat in n_features_list:
        try:
            # Create polynomial features
            x_poly = create_polynomial_features(x, n_feat)
            
            # Split data into train/test (75%/25%)
            x_train, x_test, y_train, y_test = train_test_split(
                x_poly, y, test_size=0.25, random_state=42
            )
            
            # Fit model on full sample (WITHOUT intercept as requested)
            model_full = LinearRegression(fit_intercept=False)
            model_full.fit(x_poly, y)
            y_pred_full = model_full.predict(x_poly)
            r2_full = r2_score(y, y_pred_full)
            
            # Calculate adjusted R²
            adj_r2_full = calculate_adjusted_r2(r2_full, len(y), n_feat)
            
            # Fit model on training data and predict on test data (WITHOUT intercept)
            model_train = LinearRegression(fit_intercept=False)
            model_train.fit(x_train, y_train)
            y_pred_test = model_train.predict(x_test)
            r2_out_of_sample = r2_score(y_test, y_pred_test)
            
            # Store results
            results.append({
                'n_features': n_feat,
                'r2_full': r2_full,
                'adj_r2_full': adj_r2_full,
                'r2_out_of_sample': r2_out_of_sample
            })
            
            print(f"{n_feat:8d} | {r2_full:9.4f} | {adj_r2_full:12.4f} | {r2_out_of_sample:17.4f}")
            
        except Exception as e:
            print(f"Error with {n_feat} features: {str(e)}")
            # Still append to maintain consistency
            results.append({
                'n_features': n_feat,
                'r2_full': np.nan,
                'adj_r2_full': np.nan,
                'r2_out_of_sample': np.nan
            })
    
    return pd.DataFrame(results)


def create_plots(results_df):
    """
    Create three separate plots for R-squared analysis.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing overfitting analysis results
    """
    # Filter out NaN values for plotting
    df_clean = results_df.dropna()
    
    if df_clean.empty:
        print("No valid results to plot")
        return None
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: R-squared (full sample)
    axes[0].plot(df_clean['n_features'], df_clean['r2_full'], 
                marker='o', linewidth=2, markersize=6, color='blue')
    axes[0].set_title('R-squared on Full Sample vs Number of Features', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Features')
    axes[0].set_ylabel('R-squared')
    axes[0].set_xscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Plot 2: Adjusted R-squared (full sample)
    axes[1].plot(df_clean['n_features'], df_clean['adj_r2_full'], 
                marker='s', linewidth=2, markersize=6, color='green')
    axes[1].set_title('Adjusted R-squared on Full Sample vs Number of Features', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Number of Features')
    axes[1].set_ylabel('Adjusted R-squared')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Out-of-sample R-squared
    axes[2].plot(df_clean['n_features'], df_clean['r2_out_of_sample'], 
                marker='^', linewidth=2, markersize=6, color='red')
    axes[2].set_title('Out-of-Sample R-squared vs Number of Features', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Number of Features')
    axes[2].set_ylabel('Out-of-Sample R-squared')
    axes[2].set_xscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = '/home/runner/work/High_Dimensional_Linear_Models/High_Dimensional_Linear_Models/Python/output'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/overfitting_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def interpret_results(results_df):
    """
    Interpret and summarize the overfitting analysis results.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing overfitting analysis results
    """
    print("\n=== COMPLETE RESULTS TABLE ===")
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Find optimal complexity
    valid_results = results_df.dropna()
    if not valid_results.empty:
        optimal_adj_r2_idx = valid_results['adj_r2_full'].idxmax()
        optimal_oos_r2_idx = valid_results['r2_out_of_sample'].idxmax()
        
        print("\n=== OPTIMAL MODEL COMPLEXITY ===")
        print(f"By Adjusted R²: {valid_results.loc[optimal_adj_r2_idx, 'n_features']} features")
        print(f"  - Adjusted R² = {valid_results.loc[optimal_adj_r2_idx, 'adj_r2_full']:.4f}")
        print(f"By Out-of-Sample R²: {valid_results.loc[optimal_oos_r2_idx, 'n_features']} features")
        print(f"  - Out-of-Sample R² = {valid_results.loc[optimal_oos_r2_idx, 'r2_out_of_sample']:.4f}")

    print("\n=== INSIGHTS ===")
    print("✅ This analysis demonstrates the classic bias-variance tradeoff")
    print("📈 R² (Full Sample) should increase monotonically with model complexity")
    print("📊 Adjusted R² should peak early and then decline due to complexity penalty")
    print("📉 Out-of-Sample R² should show the inverted U-shape characteristic of overfitting")
    print("🎯 True model follows: y = 2*x + e (no intercept)")
    print("⚠️ High-dimensional models (many features) lead to severe overfitting")
    
    # Save results
    output_dir = '/home/runner/work/High_Dimensional_Linear_Models/High_Dimensional_Linear_Models/Python/output'
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f'{output_dir}/overfitting_results.csv', index=False)
    print(f"\n📄 Results saved to {output_dir}/overfitting_results.csv")


def main():
    """
    Main function to run the complete overfitting analysis.
    """
    print("=" * 80)
    print("PART 2: OVERFITTING ANALYSIS")
    print("Following assignment specification: y = 2*x + e (no intercept)")
    print("=" * 80)
    
    # Run the analysis
    results_df = overfitting_analysis()
    
    # Create plots
    create_plots(results_df)
    
    # Interpret results
    interpret_results(results_df)
    
    print("\n🎉 Overfitting analysis complete!")


if __name__ == "__main__":
    main()