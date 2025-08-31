#!/usr/bin/env Rscript

# Assignment 1 - Part 1: Frisch-Waugh-Lovell (FWL) Theorem
# Math (3 points)
#
# This module contains the mathematical proof and numerical verification 
# of the Frisch-Waugh-Lovell theorem.
#
# Author: R implementation for gsaco/High_Dimensional_Linear_Models

library(MASS)  # For matrix operations

fwl_theorem_proof <- function() {
  #' Mathematical proof of the Frisch-Waugh-Lovell theorem.
  #'
  #' The FWL theorem states that the OLS estimate of β₁ in the regression of 
  #' y on [X₁ X₂] is equal to the OLS estimate obtained from the following 
  #' two-step procedure:
  #'
  #' 1. Regress y on X₂ and obtain the residuals ỹ = M_{X₂}y, 
  #'    where M_{X₂} = I - X₂(X₂'X₂)⁻¹X₂'
  #' 2. Regress X₁ on X₂ and obtain the residuals X̃₁ = M_{X₂}X₁
  #' 3. Regress ỹ on X̃₁ and show that the resulting coefficient vector 
  #'    is equal to β̂₁ from the full regression.
  #'
  #' Formally, we need to show that:
  #' β̂₁ = (X̃₁'X̃₁)⁻¹X̃₁'ỹ
  
  cat("=== FRISCH-WAUGH-LOVELL THEOREM PROOF ===\n\n")
  
  cat("Mathematical Proof:\n")
  cat("==================\n")
  cat("\n")
  cat("Consider the linear regression model:\n")
  cat("y = X₁β₁ + X₂β₂ + u\n")
  cat("\n")
  cat("Where:\n")
  cat("- y is an n×1 vector of outcomes\n")
  cat("- X₁ is an n×k₁ matrix of regressors of interest\n")
  cat("- X₂ is an n×k₂ matrix of control variables\n")
  cat("- u is an n×1 vector of errors\n")
  cat("\n")
  
  cat("Step 1: Full regression\n")
  cat("The full regression in matrix form is:\n")
  cat("y = [X₁ X₂][β₁; β₂] + u = Xβ + u\n")
  cat("\n")
  cat("The OLS estimator is:\n")
  cat("β̂ = (X'X)⁻¹X'y\n")
  cat("\n")
  cat("Partitioning X'X and X'y:\n")
  cat("X'X = [X₁'X₁  X₁'X₂]\n")
  cat("      [X₂'X₁  X₂'X₂]\n")
  cat("\n")
  cat("X'y = [X₁'y]\n")
  cat("      [X₂'y]\n")
  cat("\n")
  
  cat("Step 2: Using the partitioned inverse formula\n")
  cat("For a partitioned matrix [A B; C D], if D is invertible:\n")
  cat("The (1,1) block of the inverse is (A - BD⁻¹C)⁻¹\n")
  cat("\n")
  cat("Applying this to our case:\n")
  cat("β̂₁ = [(X₁'X₁ - X₁'X₂(X₂'X₂)⁻¹X₂'X₁)]⁻¹[X₁'y - X₁'X₂(X₂'X₂)⁻¹X₂'y]\n")
  cat("\n")
  
  cat("Step 3: Factoring out the projection matrix\n")
  cat("Let M_{X₂} = I - X₂(X₂'X₂)⁻¹X₂' (the annihilator matrix)\n")
  cat("Note that M_{X₂} is idempotent: M_{X₂}M_{X₂} = M_{X₂}\n")
  cat("And symmetric: M_{X₂}' = M_{X₂}\n")
  cat("\n")
  cat("Then:\n")
  cat("X₁'X₁ - X₁'X₂(X₂'X₂)⁻¹X₂'X₁ = X₁'[I - X₂(X₂'X₂)⁻¹X₂']X₁ = X₁'M_{X₂}X₁\n")
  cat("X₁'y - X₁'X₂(X₂'X₂)⁻¹X₂'y = X₁'[I - X₂(X₂'X₂)⁻¹X₂']y = X₁'M_{X₂}y\n")
  cat("\n")
  
  cat("Step 4: Final form\n")
  cat("Therefore:\n")
  cat("β̂₁ = (X₁'M_{X₂}X₁)⁻¹X₁'M_{X₂}y\n")
  cat("\n")
  cat("Let X̃₁ = M_{X₂}X₁ and ỹ = M_{X₂}y\n")
  cat("Then: β̂₁ = (X̃₁'X̃₁)⁻¹X̃₁'ỹ\n")
  cat("\n")
  cat("This shows that β̂₁ from the full regression equals the OLS coefficient\n")
  cat("from regressing the residuals ỹ on the residuals X̃₁.\n")
  cat("\n")
  cat("Q.E.D.\n")
  cat("\n")
}

numerical_verification <- function() {
  #' Numerical verification of the FWL theorem using simulated data.
  
  cat("=== NUMERICAL VERIFICATION ===\n\n")
  
  # Set random seed for reproducibility
  set.seed(42)
  
  # Generate data
  n <- 1000  # Sample size
  k1 <- 2    # Number of variables of interest
  k2 <- 3    # Number of control variables
  
  # Generate X1, X2, and error term
  X1 <- matrix(rnorm(n * k1), nrow = n, ncol = k1)
  X2 <- matrix(rnorm(n * k2), nrow = n, ncol = k2)
  u <- matrix(rnorm(n), nrow = n, ncol = 1)
  
  # True parameters
  beta1_true <- matrix(c(1.5, 2.0), nrow = k1, ncol = 1)
  beta2_true <- matrix(c(0.5, -1.0, 0.8), nrow = k2, ncol = 1)
  
  # Generate y
  y <- X1 %*% beta1_true + X2 %*% beta2_true + u
  
  cat(sprintf("Sample size: %d\n", n))
  cat(sprintf("X1 dimensions: (%d, %d) (variables of interest)\n", nrow(X1), ncol(X1)))
  cat(sprintf("X2 dimensions: (%d, %d) (control variables)\n", nrow(X2), ncol(X2)))
  cat(sprintf("True β₁: [%.1f, %.1f]\n", beta1_true[1], beta1_true[2]))
  cat(sprintf("True β₂: [%.1f, %.1f, %.1f]\n", beta2_true[1], beta2_true[2], beta2_true[3]))
  cat("\n")
  
  # Method 1: Full regression
  X_full <- cbind(X1, X2)
  beta_full <- solve(t(X_full) %*% X_full) %*% (t(X_full) %*% y)
  beta1_full <- beta_full[1:k1, , drop = FALSE]
  
  cat("Method 1: Full regression\n")
  cat(sprintf("β̂₁ from full regression: [%.6f, %.6f]\n", beta1_full[1], beta1_full[2]))
  cat("\n")
  
  # Method 2: FWL two-step procedure
  
  # Step 1: Regress y on X2 and get residuals
  P_X2 <- X2 %*% solve(t(X2) %*% X2) %*% t(X2)
  M_X2 <- diag(n) - P_X2
  y_tilde <- M_X2 %*% y
  
  # Step 2: Regress X1 on X2 and get residuals
  X1_tilde <- M_X2 %*% X1
  
  # Step 3: Regress y_tilde on X1_tilde
  beta1_fwl <- solve(t(X1_tilde) %*% X1_tilde) %*% (t(X1_tilde) %*% y_tilde)
  
  cat("Method 2: FWL two-step procedure\n")
  cat("Step 1: Residualize y on X₂\n")
  cat("Step 2: Residualize X₁ on X₂\n")
  cat("Step 3: Regress residuals\n")
  cat(sprintf("β̂₁ from FWL method: [%.6f, %.6f]\n", beta1_fwl[1], beta1_fwl[2]))
  cat("\n")
  
  # Check if they are equal (within numerical precision)
  difference <- abs(beta1_full - beta1_fwl)
  max_diff <- max(difference)
  
  cat("Verification:\n")
  cat(sprintf("Maximum absolute difference: %.2e\n", max_diff))
  cat(sprintf("Are they equal (within 1e-10)? %s\n", max_diff < 1e-10))
  cat("\n")
  
  # Alternative verification using lm()
  cat("Alternative verification using lm():\n")
  
  # Create data frame for lm()
  df_full <- data.frame(y = as.vector(y), X_full)
  colnames(df_full) <- c("y", paste0("X1_", 1:k1), paste0("X2_", 1:k2))
  
  # Full regression with lm()
  reg_full <- lm(y ~ . - 1, data = df_full)  # -1 removes intercept
  beta1_lm_full <- coef(reg_full)[1:k1]
  
  # FWL with lm()
  # Step 1: Get residuals y_tilde
  df_x2 <- data.frame(X2)
  colnames(df_x2) <- paste0("X2_", 1:k2)
  df_y_x2 <- data.frame(y = as.vector(y), df_x2)
  
  reg_y_on_x2 <- lm(y ~ . - 1, data = df_y_x2)
  y_tilde_lm <- residuals(reg_y_on_x2)
  
  # Step 2: Get residuals X1_tilde
  X1_tilde_lm <- matrix(0, nrow = n, ncol = k1)
  for (i in 1:k1) {
    df_x1i_x2 <- data.frame(x1i = X1[, i], df_x2)
    reg_x1i_on_x2 <- lm(x1i ~ . - 1, data = df_x1i_x2)
    X1_tilde_lm[, i] <- residuals(reg_x1i_on_x2)
  }
  
  # Step 3: Final regression
  df_fwl <- data.frame(y_tilde = y_tilde_lm, X1_tilde_lm)
  colnames(df_fwl) <- c("y_tilde", paste0("X1_tilde_", 1:k1))
  
  reg_fwl <- lm(y_tilde ~ . - 1, data = df_fwl)
  beta1_lm_fwl <- coef(reg_fwl)
  
  cat(sprintf("β̂₁ from lm() full regression: [%.6f, %.6f]\n", beta1_lm_full[1], beta1_lm_full[2]))
  cat(sprintf("β̂₁ from lm() FWL method: [%.6f, %.6f]\n", beta1_lm_fwl[1], beta1_lm_fwl[2]))
  
  diff_lm <- abs(beta1_lm_full - beta1_lm_fwl)
  max_diff_lm <- max(diff_lm)
  cat(sprintf("Maximum absolute difference (lm): %.2e\n", max_diff_lm))
  cat(sprintf("Are they equal (within 1e-10)? %s\n", max_diff_lm < 1e-10))
  
  return(list(
    beta1_full = beta1_full,
    beta1_fwl = beta1_fwl,
    beta1_lm_full = beta1_lm_full,
    beta1_lm_fwl = beta1_lm_fwl,
    max_difference = max_diff,
    max_difference_lm = max_diff_lm
  ))
}

# Main execution when run as script
if (!interactive()) {
  # Run the proof and numerical verification
  fwl_theorem_proof()
  results <- numerical_verification()
}