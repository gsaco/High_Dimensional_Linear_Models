"""
Assignment 1 - Part 1: Frisch-Waugh-Lovell (FWL) Theorem
Math (3 points)

This module contains the mathematical proof and numerical verification 
of the Frisch-Waugh-Lovell theorem.

Author: Julia implementation for gsaco/High_Dimensional_Linear_Models
"""

using LinearAlgebra
using Random
using Printf

function fwl_theorem_proof()
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
    println("=== FRISCH-WAUGH-LOVELL THEOREM PROOF ===\n")
    
    println("Mathematical Proof:")
    println("==================")
    println()
    println("Consider the linear regression model:")
    println("y = X₁β₁ + X₂β₂ + u")
    println()
    println("Where:")
    println("- y is an n×1 vector of outcomes")
    println("- X₁ is an n×k₁ matrix of regressors of interest")
    println("- X₂ is an n×k₂ matrix of control variables")
    println("- u is an n×1 vector of errors")
    println()
    
    println("Step 1: Full regression")
    println("The full regression in matrix form is:")
    println("y = [X₁ X₂][β₁; β₂] + u = Xβ + u")
    println()
    println("The OLS estimator is:")
    println("β̂ = (X'X)⁻¹X'y")
    println()
    println("Partitioning X'X and X'y:")
    println("X'X = [X₁'X₁  X₁'X₂]")
    println("      [X₂'X₁  X₂'X₂]")
    println()
    println("X'y = [X₁'y]")
    println("      [X₂'y]")
    println()
    
    println("Step 2: Using the partitioned inverse formula")
    println("For a partitioned matrix [A B; C D], if D is invertible:")
    println("The (1,1) block of the inverse is (A - BD⁻¹C)⁻¹")
    println()
    println("Applying this to our case:")
    println("β̂₁ = [(X₁'X₁ - X₁'X₂(X₂'X₂)⁻¹X₂'X₁)]⁻¹[X₁'y - X₁'X₂(X₂'X₂)⁻¹X₂'y]")
    println()
    
    println("Step 3: Factoring out the projection matrix")
    println("Let M_{X₂} = I - X₂(X₂'X₂)⁻¹X₂' (the annihilator matrix)")
    println("Note that M_{X₂} is idempotent: M_{X₂}M_{X₂} = M_{X₂}")
    println("And symmetric: M_{X₂}' = M_{X₂}")
    println()
    println("Then:")
    println("X₁'X₁ - X₁'X₂(X₂'X₂)⁻¹X₂'X₁ = X₁'[I - X₂(X₂'X₂)⁻¹X₂']X₁ = X₁'M_{X₂}X₁")
    println("X₁'y - X₁'X₂(X₂'X₂)⁻¹X₂'y = X₁'[I - X₂(X₂'X₂)⁻¹X₂']y = X₁'M_{X₂}y")
    println()
    
    println("Step 4: Final form")
    println("Therefore:")
    println("β̂₁ = (X₁'M_{X₂}X₁)⁻¹X₁'M_{X₂}y")
    println()
    println("Let X̃₁ = M_{X₂}X₁ and ỹ = M_{X₂}y")
    println("Then: β̂₁ = (X̃₁'X̃₁)⁻¹X̃₁'ỹ")
    println()
    println("This shows that β̂₁ from the full regression equals the OLS coefficient")
    println("from regressing the residuals ỹ on the residuals X̃₁.")
    println()
    println("Q.E.D.")
    println()
end

function numerical_verification()
    """
    Numerical verification of the FWL theorem using simulated data.
    """
    println("=== NUMERICAL VERIFICATION ===\n")
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Generate data
    n = 1000  # Sample size
    k1 = 2    # Number of variables of interest
    k2 = 3    # Number of control variables
    
    # Generate X1, X2, and error term
    X1 = randn(n, k1)
    X2 = randn(n, k2)
    u = randn(n, 1)
    
    # True parameters
    beta1_true = [1.5; 2.0]
    beta2_true = [0.5; -1.0; 0.8]
    
    # Generate y
    y = X1 * beta1_true + X2 * beta2_true + u
    
    @printf("Sample size: %d\n", n)
    @printf("X1 dimensions: (%d, %d) (variables of interest)\n", size(X1)...)
    @printf("X2 dimensions: (%d, %d) (control variables)\n", size(X2)...)
    @printf("True β₁: [%.1f, %.1f]\n", beta1_true...)
    @printf("True β₂: [%.1f, %.1f, %.1f]\n", beta2_true...)
    println()
    
    # Method 1: Full regression
    X_full = hcat(X1, X2)
    beta_full = (X_full' * X_full) \ (X_full' * y)
    beta1_full = beta_full[1:k1]
    
    println("Method 1: Full regression")
    @printf("β̂₁ from full regression: [%.6f, %.6f]\n", beta1_full...)
    println()
    
    # Method 2: FWL two-step procedure
    
    # Step 1: Regress y on X2 and get residuals
    P_X2 = X2 * inv(X2' * X2) * X2'
    M_X2 = I - P_X2
    y_tilde = M_X2 * y
    
    # Step 2: Regress X1 on X2 and get residuals
    X1_tilde = M_X2 * X1
    
    # Step 3: Regress y_tilde on X1_tilde
    beta1_fwl = (X1_tilde' * X1_tilde) \ (X1_tilde' * y_tilde)
    
    println("Method 2: FWL two-step procedure")
    println("Step 1: Residualize y on X₂")
    println("Step 2: Residualize X₁ on X₂")
    println("Step 3: Regress residuals")
    @printf("β̂₁ from FWL method: [%.6f, %.6f]\n", beta1_fwl...)
    println()
    
    # Check if they are equal (within numerical precision)
    difference = abs.(beta1_full - beta1_fwl)
    max_diff = maximum(difference)
    
    println("Verification:")
    @printf("Maximum absolute difference: %.2e\n", max_diff)
    @printf("Are they equal (within 1e-10)? %s\n", max_diff < 1e-10)
    println()
    
    return Dict(
        "beta1_full" => beta1_full,
        "beta1_fwl" => beta1_fwl,
        "max_difference" => max_diff
    )
end

# Main execution when run as script
if abspath(PROGRAM_FILE) == @__FILE__
    # Run the proof and numerical verification
    fwl_theorem_proof()
    results = numerical_verification()
end