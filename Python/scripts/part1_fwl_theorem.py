"""
Assignment 1 - Part 1: Frisch-Waugh-Lovell (FWL) Theorem
Math (3 points)

This module contains the mathematical proof and numerical verification 
of the Frisch-Waugh-Lovell theorem.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def fwl_theorem_proof():
    """
    Mathematical proof of the Frisch-Waugh-Lovell theorem.
    
    The FWL theorem states that the OLS estimate of β₁ in the regression of 
    y on [X₁ X₂] is equal to the OLS estimate obtained from the following 
    two-step procedure:
    
    1. Regress y on X₂ and obtain the residuals ỹ = M_{X₂}y, 
       where M_{X₂} = I - X₂(X₂'X₂)⁻¹X₂'
    2. Regress X₁ on X₂ and obtain the residuals X̃₁ = M_{X₂}X₁
    3. Regress ỹ on X̃₁ and show that the resulting coefficient vector 
       is equal to β̂₁ from the full regression.
    
    Formally, we need to show that:
    β̂₁ = (X̃₁'X̃₁)⁻¹X̃₁'ỹ
    """
    print("=== FRISCH-WAUGH-LOVELL THEOREM PROOF ===\n")
    
    print("Mathematical Proof:")
    print("==================")
    print()
    print("Consider the linear regression model:")
    print("y = X₁β₁ + X₂β₂ + u")
    print()
    print("Where:")
    print("- y is an n×1 vector of outcomes")
    print("- X₁ is an n×k₁ matrix of regressors of interest")
    print("- X₂ is an n×k₂ matrix of control variables")
    print("- u is an n×1 vector of errors")
    print()
    
    print("Step 1: Full regression")
    print("The full regression in matrix form is:")
    print("y = [X₁ X₂][β₁; β₂] + u = Xβ + u")
    print()
    print("The OLS estimator is:")
    print("β̂ = (X'X)⁻¹X'y")
    print()
    print("Partitioning X'X and X'y:")
    print("X'X = [X₁'X₁  X₁'X₂]")
    print("      [X₂'X₁  X₂'X₂]")
    print()
    print("X'y = [X₁'y]")
    print("      [X₂'y]")
    print()
    
    print("Step 2: Using the partitioned inverse formula")
    print("For a partitioned matrix [A B; C D], if D is invertible:")
    print("The (1,1) block of the inverse is (A - BD⁻¹C)⁻¹")
    print()
    print("Applying this to our case:")
    print("β̂₁ = [(X₁'X₁ - X₁'X₂(X₂'X₂)⁻¹X₂'X₁)]⁻¹[X₁'y - X₁'X₂(X₂'X₂)⁻¹X₂'y]")
    print()
    
    print("Step 3: Factoring out the projection matrix")
    print("Let M_{X₂} = I - X₂(X₂'X₂)⁻¹X₂' (the annihilator matrix)")
    print("Note that M_{X₂} is idempotent: M_{X₂}M_{X₂} = M_{X₂}")
    print("And symmetric: M_{X₂}' = M_{X₂}")
    print()
    print("Then:")
    print("X₁'X₁ - X₁'X₂(X₂'X₂)⁻¹X₂'X₁ = X₁'[I - X₂(X₂'X₂)⁻¹X₂']X₁ = X₁'M_{X₂}X₁")
    print("X₁'y - X₁'X₂(X₂'X₂)⁻¹X₂'y = X₁'[I - X₂(X₂'X₂)⁻¹X₂']y = X₁'M_{X₂}y")
    print()
    
    print("Step 4: Final form")
    print("Therefore:")
    print("β̂₁ = (X₁'M_{X₂}X₁)⁻¹X₁'M_{X₂}y")
    print()
    print("Let X̃₁ = M_{X₂}X₁ and ỹ = M_{X₂}y")
    print("Then: β̂₁ = (X̃₁'X̃₁)⁻¹X̃₁'ỹ")
    print()
    print("This shows that β̂₁ from the full regression equals the OLS coefficient")
    print("from regressing the residuals ỹ on the residuals X̃₁.")
    print()
    print("Q.E.D.")
    print()


def numerical_verification():
    """
    Numerical verification of the FWL theorem using simulated data.
    """
    print("=== NUMERICAL VERIFICATION ===\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate data
    n = 1000  # Sample size
    k1 = 2    # Number of variables of interest
    k2 = 3    # Number of control variables
    
    # Generate X1, X2, and error term
    X1 = np.random.randn(n, k1)
    X2 = np.random.randn(n, k2)
    u = np.random.randn(n, 1)
    
    # True parameters
    beta1_true = np.array([[1.5], [2.0]])
    beta2_true = np.array([[0.5], [-1.0], [0.8]])
    
    # Generate y
    y = X1 @ beta1_true + X2 @ beta2_true + u
    
    print(f"Sample size: {n}")
    print(f"X1 dimensions: {X1.shape} (variables of interest)")
    print(f"X2 dimensions: {X2.shape} (control variables)")
    print(f"True β₁: {beta1_true.ravel()}")
    print(f"True β₂: {beta2_true.ravel()}")
    print()
    
    # Method 1: Full regression
    X_full = np.column_stack([X1, X2])
    beta_full = np.linalg.inv(X_full.T @ X_full) @ X_full.T @ y
    beta1_full = beta_full[:k1]
    
    print("Method 1: Full regression")
    print(f"β̂₁ from full regression: {beta1_full.ravel()}")
    print()
    
    # Method 2: FWL two-step procedure
    
    # Step 1: Regress y on X2 and get residuals
    P_X2 = X2 @ np.linalg.inv(X2.T @ X2) @ X2.T
    M_X2 = np.eye(n) - P_X2
    y_tilde = M_X2 @ y
    
    # Step 2: Regress X1 on X2 and get residuals
    X1_tilde = M_X2 @ X1
    
    # Step 3: Regress y_tilde on X1_tilde
    beta1_fwl = np.linalg.inv(X1_tilde.T @ X1_tilde) @ X1_tilde.T @ y_tilde
    
    print("Method 2: FWL two-step procedure")
    print(f"Step 1: Residualize y on X₂")
    print(f"Step 2: Residualize X₁ on X₂")
    print(f"Step 3: Regress residuals")
    print(f"β̂₁ from FWL method: {beta1_fwl.ravel()}")
    print()
    
    # Check if they are equal (within numerical precision)
    difference = np.abs(beta1_full - beta1_fwl)
    max_diff = np.max(difference)
    
    print("Verification:")
    print(f"Maximum absolute difference: {max_diff:.2e}")
    print(f"Are they equal (within 1e-10)? {max_diff < 1e-10}")
    print()
    
    # Alternative verification using sklearn
    print("Alternative verification using sklearn:")
    
    # Full regression with sklearn
    reg_full = LinearRegression(fit_intercept=False)
    reg_full.fit(X_full, y.ravel())
    beta1_sklearn_full = reg_full.coef_[:k1]
    
    # FWL with sklearn
    # Step 1: Get residuals y_tilde
    reg_y_on_x2 = LinearRegression(fit_intercept=False)
    reg_y_on_x2.fit(X2, y.ravel())
    y_tilde_sklearn = y.ravel() - reg_y_on_x2.predict(X2)
    
    # Step 2: Get residuals X1_tilde
    X1_tilde_sklearn = np.zeros_like(X1)
    for i in range(k1):
        reg_x1_on_x2 = LinearRegression(fit_intercept=False)
        reg_x1_on_x2.fit(X2, X1[:, i])
        X1_tilde_sklearn[:, i] = X1[:, i] - reg_x1_on_x2.predict(X2)
    
    # Step 3: Final regression
    reg_fwl = LinearRegression(fit_intercept=False)
    reg_fwl.fit(X1_tilde_sklearn, y_tilde_sklearn)
    beta1_sklearn_fwl = reg_fwl.coef_
    
    print(f"β̂₁ from sklearn full regression: {beta1_sklearn_full}")
    print(f"β̂₁ from sklearn FWL method: {beta1_sklearn_fwl}")
    
    diff_sklearn = np.abs(beta1_sklearn_full - beta1_sklearn_fwl)
    max_diff_sklearn = np.max(diff_sklearn)
    print(f"Maximum absolute difference (sklearn): {max_diff_sklearn:.2e}")
    print(f"Are they equal (within 1e-10)? {max_diff_sklearn < 1e-10}")
    
    return {
        'beta1_full': beta1_full,
        'beta1_fwl': beta1_fwl,
        'beta1_sklearn_full': beta1_sklearn_full,
        'beta1_sklearn_fwl': beta1_sklearn_fwl,
        'max_difference': max_diff,
        'max_difference_sklearn': max_diff_sklearn
    }


if __name__ == "__main__":
    # Run the proof and numerical verification
    fwl_theorem_proof()
    results = numerical_verification()