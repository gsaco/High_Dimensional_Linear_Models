#!/usr/bin/env python
# coding: utf-8

# # Assignment 1 - Part 2: Overfitting Analysis
# ## Overfitting (8 points)
# 
# This notebook simulates a data generating process and analyzes overfitting by estimating linear models with increasing numbers of polynomial features.

# ## Import Required Libraries

# In[33]:


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


# ## Data Generation
# 
# We'll generate data following an exponential relationship: y = exp(4X) + e, where e is random noise. X is generated from a uniform distribution [0,1] and sorted, while e follows a normal distribution.

# In[34]:


def generate_data(n=1000, seed=42):
    """
    Generate data following the specification with only 2 variables X and Y.
    Uses the new PGD: y = exp(4*X) + e

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
    e : numpy.ndarray
        Error term
    """
    np.random.seed(seed)

    # Generate X using uniform distribution [0,1], sorted
    X = np.random.uniform(0, 1, n)
    X = X.reshape(-1, 1)

    # Generate error term e using normal distribution
    e = np.random.normal(0, 1, n)
    e = e.reshape(-1, 1)

    # Generate y using the new PGD: y = exp(4*X) + e
    y = np.exp(4 * X.ravel()) + e.ravel()

    return X, y, e

# Generate the data
X, y, e = generate_data(n=1000, seed=42)

print(f"Generated data with n={len(y)} observations")


# ## Polynomial Feature Creation
# 
# We'll create polynomial features of increasing complexity to study overfitting behavior.

# In[35]:


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


# ## Adjusted R-squared Calculation
# 
# We'll implement adjusted R-squared, which penalizes model complexity.

# In[36]:


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


# ## Overfitting Analysis
# 
# Now we'll perform the main analysis, testing models with different numbers of polynomial features.

# In[37]:


def overfitting_analysis():
    """
    Main function to perform overfitting analysis.
    """
    print("=== OVERFITTING ANALYSIS ===\n")

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

# Run the analysis
results_df = overfitting_analysis()


# ## Visualization
# 
# Let's create three plots to visualize the different R-squared measures as a function of model complexity.

# In[38]:


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
    plt.show()

    print("Plots created successfully!")

# Create the plots
create_plots(results_df)


# ## Save Results
# 
# Finally, let's save our results for future reference.

# In[39]:


# Save results to CSV
import os
output_dir = '../output'  # Relative path to Python/output directory
os.makedirs(output_dir, exist_ok=True)

results_df.to_csv(f'{output_dir}/overfitting_results.csv', index=False)
print(f"Results saved to {output_dir}/overfitting_results.csv")

# Also save summary statistics
summary_stats.to_csv(f'{output_dir}/overfitting_summary.csv', index=False)
print(f"Summary statistics saved to {output_dir}/overfitting_summary.csv")

