"""
Assignment 1 - Part 2: Overfitting Analysis
Overfitting (8 points)

This module simulates a data generating process and analyzes overfitting
by estimating linear models with increasing numbers of features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


def generate_data(n=1000, seed=42):
    """
    Generate data following the specification in Lab2 with only 2 variables X and Y.
    Intercept parameter is set to zero as requested.
    
    Parameters:
    -----------
    n : int
        Sample size (default: 1000)
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target variable
    """
    np.random.seed(seed)
    
    # Generate X (single feature initially)
    X = np.random.randn(n, 1)
    
    # Generate error term
    u = np.random.randn(n)
    
    # Generate y with no intercept (as requested)
    # True relationship: y = 2*X + u
    beta_true = 2.0
    y = beta_true * X.ravel() + u
    
    return X, y


def create_polynomial_features(X, n_features):
    """
    Create polynomial features up to n_features.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Original feature matrix (n x 1)
    n_features : int
        Number of features to create
        
    Returns:
    --------
    X_poly : numpy.ndarray
        Extended feature matrix with polynomial features
    """
    n_samples = X.shape[0]
    X_poly = np.zeros((n_samples, n_features))
    
    for i in range(n_features):
        if i == 0:
            X_poly[:, i] = X.ravel()  # x^1
        else:
            X_poly[:, i] = X.ravel() ** (i + 1)  # x^2, x^3, etc.
    
    return X_poly


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
    if n - k - 1 <= 0:
        return np.nan
    
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    return adj_r2


def overfitting_analysis():
    """
    Main function to perform overfitting analysis.
    """
    print("=== OVERFITTING ANALYSIS ===\n")
    
    # Generate data
    X, y = generate_data(n=1000, seed=42)
    
    print(f"Generated data with n={len(y)} observations")
    print(f"True relationship: y = 2*X + u")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print()
    
    # Number of features to test
    n_features_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    # Storage for results
    results = {
        'n_features': [],
        'r2_full': [],
        'adj_r2_full': [],
        'r2_out_of_sample': []
    }
    
    print("Analyzing overfitting for different numbers of features...")
    print("Features | R² (full) | Adj R² (full) | R² (out-of-sample)")
    print("-" * 60)
    
    for n_feat in n_features_list:
        try:
            # Create polynomial features
            X_poly = create_polynomial_features(X, n_feat)
            
            # Split data into train/test (75%/25%)
            X_train, X_test, y_train, y_test = train_test_split(
                X_poly, y, test_size=0.25, random_state=42
            )
            
            # Fit model on full sample
            reg_full = LinearRegression(fit_intercept=False)  # No intercept as requested
            reg_full.fit(X_poly, y)
            y_pred_full = reg_full.predict(X_poly)
            r2_full = r2_score(y, y_pred_full)
            
            # Calculate adjusted R²
            adj_r2_full = calculate_adjusted_r2(r2_full, len(y), n_feat)
            
            # Fit model on training data and predict on test data
            reg_train = LinearRegression(fit_intercept=False)
            reg_train.fit(X_train, y_train)
            y_pred_test = reg_train.predict(X_test)
            r2_out_of_sample = r2_score(y_test, y_pred_test)
            
            # Store results
            results['n_features'].append(n_feat)
            results['r2_full'].append(r2_full)
            results['adj_r2_full'].append(adj_r2_full)
            results['r2_out_of_sample'].append(r2_out_of_sample)
            
            print(f"{n_feat:8d} | {r2_full:9.4f} | {adj_r2_full:12.4f} | {r2_out_of_sample:17.4f}")
            
        except Exception as e:
            print(f"Error with {n_feat} features: {e}")
            # Still append to maintain list length
            results['n_features'].append(n_feat)
            results['r2_full'].append(np.nan)
            results['adj_r2_full'].append(np.nan)
            results['r2_out_of_sample'].append(np.nan)
    
    print()
    
    # Convert to DataFrame for easier handling
    df_results = pd.DataFrame(results)
    
    return df_results


def create_plots(df_results):
    """
    Create three separate plots for R-squared analysis.
    
    Parameters:
    -----------
    df_results : pandas.DataFrame
        Results from overfitting analysis
    """
    print("Creating plots...")
    
    # Set up the plotting parameters
    fig_size = (12, 5)
    
    # Plot 1: R-squared (full sample)
    plt.figure(figsize=fig_size)
    plt.plot(df_results['n_features'], df_results['r2_full'], 
             marker='o', linewidth=2, markersize=6, color='blue')
    plt.title('R-squared on Full Sample vs Number of Features', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('R-squared', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('/home/runner/work/High_Dimensional_Linear_Models/High_Dimensional_Linear_Models/Python/output/r2_full_sample.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Adjusted R-squared (full sample)
    plt.figure(figsize=fig_size)
    plt.plot(df_results['n_features'], df_results['adj_r2_full'], 
             marker='s', linewidth=2, markersize=6, color='green')
    plt.title('Adjusted R-squared on Full Sample vs Number of Features', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Adjusted R-squared', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('/home/runner/work/High_Dimensional_Linear_Models/High_Dimensional_Linear_Models/Python/output/adj_r2_full_sample.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Out-of-sample R-squared
    plt.figure(figsize=fig_size)
    plt.plot(df_results['n_features'], df_results['r2_out_of_sample'], 
             marker='^', linewidth=2, markersize=6, color='red')
    plt.title('Out-of-Sample R-squared vs Number of Features', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Out-of-Sample R-squared', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('/home/runner/work/High_Dimensional_Linear_Models/High_Dimensional_Linear_Models/Python/output/r2_out_of_sample.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plots saved to Python/output/ directory")


def interpret_results(df_results):
    """
    Provide interpretation and intuition for the results.
    
    Parameters:
    -----------
    df_results : pandas.DataFrame
        Results from overfitting analysis
    """
    print("\n=== RESULTS INTERPRETATION ===\n")
    
    print("Key Observations:")
    print("================")
    
    # R-squared observations
    max_r2_full = df_results['r2_full'].max()
    max_r2_features = df_results.loc[df_results['r2_full'].idxmax(), 'n_features']
    
    print(f"1. R-squared (Full Sample):")
    print(f"   - Starts at {df_results['r2_full'].iloc[0]:.4f} with 1 feature")
    print(f"   - Reaches maximum of {max_r2_full:.4f} with {max_r2_features} features")
    print(f"   - Shows monotonic increase as expected in in-sample fit")
    print()
    
    # Adjusted R-squared observations
    max_adj_r2 = df_results['adj_r2_full'].max()
    max_adj_r2_features = df_results.loc[df_results['adj_r2_full'].idxmax(), 'n_features']
    
    print(f"2. Adjusted R-squared (Full Sample):")
    print(f"   - Peaks at {max_adj_r2:.4f} with {max_adj_r2_features} features")
    print(f"   - Then declines as the penalty for additional features outweighs benefit")
    print(f"   - Becomes negative when model is severely overfitted")
    print()
    
    # Out-of-sample observations
    max_oos_r2 = df_results['r2_out_of_sample'].max()
    max_oos_r2_features = df_results.loc[df_results['r2_out_of_sample'].idxmax(), 'n_features']
    min_oos_r2 = df_results['r2_out_of_sample'].min()
    
    print(f"3. Out-of-Sample R-squared:")
    print(f"   - Peaks at {max_oos_r2:.4f} with {max_oos_r2_features} features")
    print(f"   - Drops dramatically to {min_oos_r2:.4f} as overfitting increases")
    print(f"   - Can become negative when predictions are worse than using the mean")
    print()
    
    print("Economic Intuition:")
    print("==================")
    print()
    print("1. **Bias-Variance Tradeoff**: As we add more features (higher-order polynomials),")
    print("   we reduce bias but increase variance. Initially, bias reduction dominates,")
    print("   improving out-of-sample performance. Eventually, variance dominates.")
    print()
    print("2. **In-Sample vs Out-of-Sample**: In-sample R² always increases with more features")
    print("   because the model can always fit the training data better. However, this")
    print("   doesn't translate to better prediction on new data.")
    print()
    print("3. **Adjusted R-squared as a Model Selection Tool**: Adjusted R² penalizes model")
    print("   complexity and provides a better guide for model selection than raw R².")
    print()
    print("4. **The Curse of Dimensionality**: With 1000 observations and up to 1000 features,")
    print("   we approach the case where we have as many parameters as observations,")
    print("   leading to perfect in-sample fit but terrible out-of-sample performance.")
    print()
    print("5. **Practical Implications**: This demonstrates why regularization techniques")
    print("   (Ridge, Lasso, Elastic Net) are crucial in high-dimensional settings to")
    print("   prevent overfitting and improve generalization.")


if __name__ == "__main__":
    # Run overfitting analysis
    results_df = overfitting_analysis()
    
    # Create plots
    create_plots(results_df)
    
    # Interpret results
    interpret_results(results_df)
    
    # Save results to CSV
    results_df.to_csv('/home/runner/work/High_Dimensional_Linear_Models/High_Dimensional_Linear_Models/Python/output/overfitting_results.csv', 
                      index=False)
    print(f"\nResults saved to Python/output/overfitting_results.csv")