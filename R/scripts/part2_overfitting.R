#!/usr/bin/env Rscript

# Assignment 1 - Part 2: Overfitting Analysis
# Overfitting (8 points)
#
# This module simulates a data generating process and analyzes overfitting
# by estimating linear models with increasing numbers of features.
#
# Author: R implementation for gsaco/High_Dimensional_Linear_Models

library(ggplot2)
library(dplyr)

generate_data <- function(n = 1000, seed = 42) {
  #' Generate data following the specification in Lab2 with only 2 variables X and Y.
  #' Intercept parameter is set to zero as requested.
  #'
  #' @param n Sample size (default: 1000)
  #' @param seed Random seed for reproducibility
  #'
  #' @return List containing X (feature matrix) and y (target variable)
  
  set.seed(seed)
  
  # Generate X (single feature initially)
  X <- matrix(rnorm(n), nrow = n, ncol = 1)
  
  # Generate error term
  u <- rnorm(n)
  
  # Generate y with no intercept (as requested)
  # True relationship: y = 2*X + u
  beta_true <- 2.0
  y <- beta_true * X[, 1] + u
  
  return(list(X = X, y = y))
}

create_polynomial_features <- function(X, n_features) {
  #' Create polynomial features up to n_features.
  #'
  #' @param X Original feature matrix (n x 1)
  #' @param n_features Number of features to create
  #'
  #' @return Extended feature matrix with polynomial features
  
  n_samples <- nrow(X)
  X_poly <- matrix(0, nrow = n_samples, ncol = n_features)
  
  for (i in 1:n_features) {
    X_poly[, i] <- X[, 1]^i  # x^1, x^2, x^3, etc.
  }
  
  return(X_poly)
}

calculate_adjusted_r2 <- function(r2, n, k) {
  #' Calculate adjusted R-squared.
  #'
  #' Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - k - 1)]
  #'
  #' @param r2 R-squared value
  #' @param n Sample size
  #' @param k Number of features (excluding intercept)
  #'
  #' @return Adjusted R-squared
  
  if (n - k - 1 <= 0) {
    return(NA)
  }
  
  adj_r2 <- 1 - ((1 - r2) * (n - 1) / (n - k - 1))
  return(adj_r2)
}

r2_score <- function(y_true, y_pred) {
  #' Calculate R-squared score.
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  return(1 - (ss_res / ss_tot))
}

train_test_split <- function(X, y, test_size = 0.25, random_state = 42) {
  #' Split data into training and testing sets.
  set.seed(random_state)
  n <- length(y)
  n_test <- round(n * test_size)
  indices <- sample(1:n, n)
  
  test_indices <- indices[1:n_test]
  train_indices <- indices[(n_test + 1):n]
  
  return(list(
    X_train = X[train_indices, , drop = FALSE],
    X_test = X[test_indices, , drop = FALSE],
    y_train = y[train_indices],
    y_test = y[test_indices]
  ))
}

overfitting_analysis <- function() {
  #' Main function to perform overfitting analysis.
  
  cat("=== OVERFITTING ANALYSIS ===\n\n")
  
  # Generate data
  data <- generate_data(n = 1000, seed = 42)
  X <- data$X
  y <- data$y
  
  cat(sprintf("Generated data with n=%d observations\n", length(y)))
  cat("True relationship: y = 2*X + u\n")
  cat(sprintf("X shape: (%d, %d)\n", nrow(X), ncol(X)))
  cat(sprintf("y shape: (%d)\n", length(y)))
  cat("\n")
  
  # Number of features to test
  n_features_list <- c(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
  
  # Storage for results
  results <- data.frame(
    n_features = integer(),
    r2_full = numeric(),
    adj_r2_full = numeric(),
    r2_out_of_sample = numeric()
  )
  
  cat("Analyzing overfitting for different numbers of features...\n")
  cat("Features | R² (full) | Adj R² (full) | R² (out-of-sample)\n")
  cat(paste(rep("-", 60), collapse = ""), "\n")
  
  for (n_feat in n_features_list) {
    tryCatch({
      # Create polynomial features
      X_poly <- create_polynomial_features(X, n_feat)
      
      # Split data into train/test (75%/25%)
      split_data <- train_test_split(X_poly, y, test_size = 0.25, random_state = 42)
      X_train <- split_data$X_train
      X_test <- split_data$X_test
      y_train <- split_data$y_train
      y_test <- split_data$y_test
      
      # Fit model on full sample (no intercept as requested)
      beta_full <- solve(t(X_poly) %*% X_poly) %*% (t(X_poly) %*% y)
      y_pred_full <- X_poly %*% beta_full
      r2_full <- r2_score(y, y_pred_full)
      
      # Calculate adjusted R²
      adj_r2_full <- calculate_adjusted_r2(r2_full, length(y), n_feat)
      
      # Fit model on training data and predict on test data
      beta_train <- solve(t(X_train) %*% X_train) %*% (t(X_train) %*% y_train)
      y_pred_test <- X_test %*% beta_train
      r2_out_of_sample <- r2_score(y_test, y_pred_test)
      
      # Store results
      results <- rbind(results, data.frame(
        n_features = n_feat,
        r2_full = r2_full,
        adj_r2_full = adj_r2_full,
        r2_out_of_sample = r2_out_of_sample
      ))
      
      cat(sprintf("%8d | %9.4f | %12.4f | %17.4f\n", 
                  n_feat, r2_full, adj_r2_full, r2_out_of_sample))
      
    }, error = function(e) {
      cat(sprintf("Error with %d features: %s\n", n_feat, e$message))
      # Still append to maintain list length
      results <<- rbind(results, data.frame(
        n_features = n_feat,
        r2_full = NA,
        adj_r2_full = NA,
        r2_out_of_sample = NA
      ))
    })
  }
  
  cat("\n")
  return(results)
}

create_plots <- function(df_results) {
  #' Create three separate plots for R-squared analysis.
  #'
  #' @param df_results Results from overfitting analysis
  
  cat("Creating plots...\n")
  
  # Create output directory if it doesn't exist
  output_dir <- "/home/runner/work/High_Dimensional_Linear_Models/High_Dimensional_Linear_Models/R/output"
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Plot 1: R-squared (full sample)
  p1 <- ggplot(df_results, aes(x = n_features, y = r2_full)) +
    geom_line(color = "blue", size = 1) +
    geom_point(color = "blue", size = 2) +
    scale_x_log10() +
    ylim(0, 1) +
    labs(
      title = "R-squared on Full Sample vs Number of Features",
      x = "Number of Features",
      y = "R-squared"
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 10)
    )
  
  ggsave(file.path(output_dir, "r2_full_sample.png"), p1, 
         width = 10, height = 4, dpi = 300)
  
  # Plot 2: Adjusted R-squared (full sample)
  p2 <- ggplot(df_results, aes(x = n_features, y = adj_r2_full)) +
    geom_line(color = "green", size = 1) +
    geom_point(color = "green", size = 2, shape = 15) +
    scale_x_log10() +
    labs(
      title = "Adjusted R-squared on Full Sample vs Number of Features",
      x = "Number of Features",
      y = "Adjusted R-squared"
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 10)
    )
  
  ggsave(file.path(output_dir, "adj_r2_full_sample.png"), p2, 
         width = 10, height = 4, dpi = 300)
  
  # Plot 3: Out-of-sample R-squared
  p3 <- ggplot(df_results, aes(x = n_features, y = r2_out_of_sample)) +
    geom_line(color = "red", size = 1) +
    geom_point(color = "red", size = 2, shape = 17) +
    scale_x_log10() +
    labs(
      title = "Out-of-Sample R-squared vs Number of Features",
      x = "Number of Features",
      y = "Out-of-Sample R-squared"
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 10)
    )
  
  ggsave(file.path(output_dir, "r2_out_of_sample.png"), p3, 
         width = 10, height = 4, dpi = 300)
  
  cat("Plots saved to R/output/ directory\n")
}

interpret_results <- function(df_results) {
  #' Provide interpretation and intuition for the results.
  #'
  #' @param df_results Results from overfitting analysis
  
  cat("\n=== RESULTS INTERPRETATION ===\n\n")
  
  cat("Key Observations:\n")
  cat("================\n")
  
  # R-squared observations
  max_r2_full <- max(df_results$r2_full, na.rm = TRUE)
  max_r2_idx <- which.max(df_results$r2_full)
  max_r2_features <- df_results$n_features[max_r2_idx]
  
  cat(sprintf("1. R-squared (Full Sample):\n"))
  cat(sprintf("   - Starts at %.4f with 1 feature\n", df_results$r2_full[1]))
  cat(sprintf("   - Reaches maximum of %.4f with %d features\n", max_r2_full, max_r2_features))
  cat(sprintf("   - Shows monotonic increase as expected in in-sample fit\n"))
  cat("\n")
  
  # Adjusted R-squared observations
  valid_adj_r2 <- df_results$adj_r2_full[!is.na(df_results$adj_r2_full)]
  if (length(valid_adj_r2) > 0) {
    max_adj_r2 <- max(valid_adj_r2)
    max_adj_r2_idx <- which.max(df_results$adj_r2_full)
    max_adj_r2_features <- df_results$n_features[max_adj_r2_idx]
    
    cat(sprintf("2. Adjusted R-squared (Full Sample):\n"))
    cat(sprintf("   - Peaks at %.4f with %d features\n", max_adj_r2, max_adj_r2_features))
    cat(sprintf("   - Then declines as the penalty for additional features outweighs benefit\n"))
    cat(sprintf("   - Becomes negative when model is severely overfitted\n"))
    cat("\n")
  }
  
  # Out-of-sample observations
  valid_oos_r2 <- df_results$r2_out_of_sample[!is.na(df_results$r2_out_of_sample)]
  if (length(valid_oos_r2) > 0) {
    max_oos_r2 <- max(valid_oos_r2)
    max_oos_r2_idx <- which.max(df_results$r2_out_of_sample)
    max_oos_r2_features <- df_results$n_features[max_oos_r2_idx]
    min_oos_r2 <- min(valid_oos_r2)
    
    cat(sprintf("3. Out-of-Sample R-squared:\n"))
    cat(sprintf("   - Peaks at %.4f with %d features\n", max_oos_r2, max_oos_r2_features))
    cat(sprintf("   - Drops dramatically to %.4f as overfitting increases\n", min_oos_r2))
    cat(sprintf("   - Can become negative when predictions are worse than using the mean\n"))
    cat("\n")
  }
  
  cat("Economic Intuition:\n")
  cat("==================\n")
  cat("\n")
  cat("1. **Bias-Variance Tradeoff**: As we add more features (higher-order polynomials),\n")
  cat("   we reduce bias but increase variance. Initially, bias reduction dominates,\n")
  cat("   improving out-of-sample performance. Eventually, variance dominates.\n")
  cat("\n")
  cat("2. **In-Sample vs Out-of-Sample**: In-sample R² always increases with more features\n")
  cat("   because the model can always fit the training data better. However, this\n")
  cat("   doesn't translate to better prediction on new data.\n")
  cat("\n")
  cat("3. **Adjusted R-squared as a Model Selection Tool**: Adjusted R² penalizes model\n")
  cat("   complexity and provides a better guide for model selection than raw R².\n")
  cat("\n")
  cat("4. **The Curse of Dimensionality**: With 1000 observations and up to 1000 features,\n")
  cat("   we approach the case where we have as many parameters as observations,\n")
  cat("   leading to perfect in-sample fit but terrible out-of-sample performance.\n")
  cat("\n")
  cat("5. **Practical Implications**: This demonstrates why regularization techniques\n")
  cat("   (Ridge, Lasso, Elastic Net) are crucial in high-dimensional settings to\n")
  cat("   prevent overfitting and improve generalization.\n")
}

# Main execution when run as script
if (!interactive()) {
  # Run overfitting analysis
  results_df <- overfitting_analysis()
  
  # Create plots
  create_plots(results_df)
  
  # Interpret results
  interpret_results(results_df)
  
  # Save results to CSV
  output_dir <- "/home/runner/work/High_Dimensional_Linear_Models/High_Dimensional_Linear_Models/R/output"
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  write.csv(results_df, file.path(output_dir, "overfitting_results.csv"), row.names = FALSE)
  cat("\nResults saved to R/output/overfitting_results.csv\n")
}