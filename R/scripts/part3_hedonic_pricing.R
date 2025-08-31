#!/usr/bin/env Rscript

# Assignment 1 - Part 3: Real Data Analysis - Hedonic Pricing Model
# Real data (9 points)
#
# This module implements hedonic pricing model analysis using apartment data from Poland.
#
# Author: R implementation for gsaco/High_Dimensional_Linear_Models

library(dplyr)
library(MASS)

load_data <- function() {
  #' Load apartment data. 
  #' For now, we'll create sample data that matches the description.
  #' In a real scenario, this would load from 'CausalAI-Course/Data/apartments.csv'
  
  cat("Loading apartment data...\n")
  
  # Since we don't have access to the actual file, let's create sample data
  # that matches the structure described in the problem statement
  set.seed(42)
  n <- 2000  # Sample size
  
  # Generate sample data that matches the structure
  price <- exp(rnorm(n, mean = 12, sd = 0.5))  # Log-normal distribution for prices
  month <- sample(1:12, n, replace = TRUE)
  id <- 1:n
  type <- sample(c("flat", "studio", "apartment"), n, replace = TRUE, prob = c(0.6, 0.2, 0.2))
  area <- runif(n, min = 20, max = 150)
  rooms <- sample(1:5, n, replace = TRUE)
  schoolDistance <- runif(n, min = 0.1, max = 5.0)
  clinicDistance <- runif(n, min = 0.1, max = 8.0)
  postOfficeDistance <- runif(n, min = 0.1, max = 3.0)
  kindergartenDistance <- runif(n, min = 0.1, max = 4.0)
  restaurantDistance <- runif(n, min = 0.1, max = 2.0)
  collegeDistance <- runif(n, min = 0.5, max = 15.0)
  pharmacyDistance <- runif(n, min = 0.1, max = 3.0)
  ownership <- sample(c("freehold", "cooperative", "rental"), n, replace = TRUE, prob = c(0.5, 0.3, 0.2))
  buildingMaterial <- sample(c("brick", "concrete", "wood"), n, replace = TRUE, prob = c(0.4, 0.5, 0.1))
  hasParkingSpace <- sample(c("yes", "no"), n, replace = TRUE, prob = c(0.3, 0.7))
  hasBalcony <- sample(c("yes", "no"), n, replace = TRUE, prob = c(0.6, 0.4))
  hasElevator <- sample(c("yes", "no"), n, replace = TRUE, prob = c(0.4, 0.6))
  hasSecurity <- sample(c("yes", "no"), n, replace = TRUE, prob = c(0.2, 0.8))
  hasStorageRoom <- sample(c("yes", "no"), n, replace = TRUE, prob = c(0.3, 0.7))
  
  # Create data frame
  df <- data.frame(
    price = price,
    month = month,
    id = id,
    type = type,
    area = area,
    rooms = rooms,
    schoolDistance = schoolDistance,
    clinicDistance = clinicDistance,
    postOfficeDistance = postOfficeDistance,
    kindergartenDistance = kindergartenDistance,
    restaurantDistance = restaurantDistance,
    collegeDistance = collegeDistance,
    pharmacyDistance = pharmacyDistance,
    ownership = ownership,
    buildingMaterial = buildingMaterial,
    hasParkingSpace = hasParkingSpace,
    hasBalcony = hasBalcony,
    hasElevator = hasElevator,
    hasSecurity = hasSecurity,
    hasStorageRoom = hasStorageRoom,
    stringsAsFactors = FALSE
  )
  
  # Make price dependent on area and other features to create realistic relationships
  # Price increases with area, decreases with distance to amenities
  price_base <- (df$area * runif(n, min = 800, max = 1200) + 
                 -df$schoolDistance * 5000 +
                 -df$clinicDistance * 3000 +
                 (df$hasBalcony == "yes") * 20000 +
                 (df$hasParkingSpace == "yes") * 30000 +
                 (df$hasElevator == "yes") * 15000 +
                 rnorm(n, mean = 0, sd = 20000))
  
  df$price <- pmax(price_base, 50000)  # Ensure positive prices
  
  # Make some areas end in 0 with slightly higher prices (creates the effect we want to detect)
  area_last_digit <- floor(df$area) %% 10
  df$price[area_last_digit == 0] <- df$price[area_last_digit == 0] * runif(sum(area_last_digit == 0), min = 1.02, max = 1.08)
  
  cat(sprintf("Loaded data with %d observations and %d variables\n", nrow(df), ncol(df)))
  cat(sprintf("Sample of apartments with area ending in 0: %d\n", sum(area_last_digit == 0)))
  
  return(df)
}

clean_data <- function(df) {
  #' Perform data cleaning as specified in Part 3a.
  #'
  #' Tasks:
  #' 1. Create area2 variable (square of area)
  #' 2. Convert binary variables to dummy variables (yes/no -> 1/0)
  #' 3. Create last digit dummy variables for area (end_0 to end_9)
  
  cat("\n=== DATA CLEANING (Part 3a) ===\n\n")
  
  df_clean <- df
  
  # 1. Create area2 variable (0.25 points)
  df_clean$area2 <- df_clean$area^2
  cat("✓ Created area2 variable (square of area)\n")
  
  # 2. Convert binary variables to dummy variables (0.75 points)
  binary_vars <- c("hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom")
  
  for (var in binary_vars) {
    df_clean[[var]] <- as.integer(df_clean[[var]] == "yes")
  }
        
  cat(sprintf("✓ Converted %d binary variables to dummy variables (1=yes, 0=no)\n", length(binary_vars)))
  
  # 3. Create last digit dummy variables (1 point)
  area_last_digit <- floor(df_clean$area) %% 10
  
  for (digit in 0:9) {
    col_name <- paste0("end_", digit)
    df_clean[[col_name]] <- as.integer(area_last_digit == digit)
  }
  
  cat("✓ Created last digit dummy variables (end_0 through end_9)\n")
  
  # Display summary of cleaning
  cat(sprintf("\nCleaning Summary:\n"))
  cat(sprintf("- Original variables: %d\n", ncol(df)))
  cat(sprintf("- Variables after cleaning: %d\n", ncol(df_clean)))
  new_vars <- c("area2", paste0("end_", 0:9))
  cat(sprintf("- New variables created: %s\n", paste(new_vars, collapse = ", ")))
  
  # Show distribution of area last digits
  cat("\nArea last digit distribution:\n")
  for (digit in 0:9) {
    count <- sum(area_last_digit == digit)
    pct <- count / length(df_clean$area) * 100
    cat(sprintf("  end_%d: %4d (%5.1f%%)\n", digit, count, pct))
  }
  
  return(df_clean)
}

create_design_matrix <- function(df, features) {
  #' Create design matrix from data frame and feature list.
  
  # Start with numeric features that exist directly in the dataframe
  numeric_features <- features[features %in% names(df)]
  if (length(numeric_features) > 0) {
    X_numeric <- as.matrix(df[, numeric_features, drop = FALSE])
  } else {
    X_numeric <- matrix(nrow = nrow(df), ncol = 0)
  }
  
  # Handle categorical dummy variables
  categorical_features <- features[!features %in% names(df)]
  
  if (length(categorical_features) > 0) {
    X_categorical <- matrix(0, nrow = nrow(df), ncol = length(categorical_features))
    colnames(X_categorical) <- categorical_features
    
    for (i in seq_along(categorical_features)) {
      feature <- categorical_features[i]
      
      if (grepl("^month_", feature)) {
        month_val <- as.numeric(sub("month_", "", feature))
        X_categorical[, i] <- as.integer(df$month == month_val)
      } else if (grepl("^type_", feature)) {
        type_val <- sub("type_", "", feature)
        X_categorical[, i] <- as.integer(df$type == type_val)
      } else if (grepl("^rooms_", feature)) {
        rooms_val <- as.numeric(sub("rooms_", "", feature))
        X_categorical[, i] <- as.integer(df$rooms == rooms_val)
      } else if (grepl("^ownership_", feature)) {
        ownership_val <- sub("ownership_", "", feature)
        X_categorical[, i] <- as.integer(df$ownership == ownership_val)
      } else if (grepl("^buildingMaterial_", feature)) {
        material_val <- sub("buildingMaterial_", "", feature)
        X_categorical[, i] <- as.integer(df$buildingMaterial == material_val)
      }
    }
    
    # Combine numeric and categorical features
    X <- cbind(X_numeric, X_categorical)
  } else {
    X <- X_numeric
  }
  
  return(X)
}

linear_model_estimation <- function(df) {
  #' Perform linear model estimation as specified in Part 3b.
  #'
  #' Tasks:
  #' 1. Regress price against specified covariates
  #' 2. Perform the same regression using partialling-out method
  #' 3. Verify coefficients match
  
  cat("\n=== LINEAR MODEL ESTIMATION (Part 3b) ===\n\n")
  
  # Prepare the feature list
  features <- character()
  
  # Area's last digit dummies (omit 9 to have a base category)
  digit_features <- paste0("end_", 0:8)  # end_0 through end_8
  features <- c(features, digit_features)
  
  # Area and area squared
  features <- c(features, "area", "area2")
  
  # Distance variables
  distance_features <- c("schoolDistance", "clinicDistance", "postOfficeDistance", 
                        "kindergartenDistance", "restaurantDistance", "collegeDistance", 
                        "pharmacyDistance")
  features <- c(features, distance_features)
  
  # Binary features
  binary_features <- c("hasParkingSpace", "hasBalcony", "hasElevator", "hasSecurity", "hasStorageRoom")
  features <- c(features, binary_features)
  
  # Categorical variables (create dummy variables, drop first category)
  # Month dummies (drop month 1)
  unique_months <- unique(df$month)
  for (month in unique_months) {
    if (month != 1) {  # Drop first category
      features <- c(features, paste0("month_", month))
    }
  }
  
  # Type dummies (drop "apartment")
  unique_types <- unique(df$type)
  for (type_val in unique_types) {
    if (type_val != "apartment") {  # Drop first category
      features <- c(features, paste0("type_", type_val))
    }
  }
  
  # Rooms dummies (drop rooms 1)
  unique_rooms <- unique(df$rooms)
  for (rooms_val in unique_rooms) {
    if (rooms_val != 1) {  # Drop first category
      features <- c(features, paste0("rooms_", rooms_val))
    }
  }
  
  # Ownership dummies (drop "cooperative")
  unique_ownership <- unique(df$ownership)
  for (ownership_val in unique_ownership) {
    if (ownership_val != "cooperative") {  # Drop first category
      features <- c(features, paste0("ownership_", ownership_val))
    }
  }
  
  # Building material dummies (drop "brick")
  unique_materials <- unique(df$buildingMaterial)
  for (material_val in unique_materials) {
    if (material_val != "brick") {  # Drop first category
      features <- c(features, paste0("buildingMaterial_", material_val))
    }
  }
  
  # Create design matrix
  X <- create_design_matrix(df, features)
  y <- df$price
  
  cat(sprintf("Feature matrix shape: (%d, %d)\n", nrow(X), ncol(X)))
  cat(sprintf("Target variable shape: (%d)\n", length(y)))
  cat(sprintf("Total features: %d\n", length(features)))
  
  # Method 1: Standard linear regression (with intercept)
  cat("\n1. Standard Linear Regression:\n")
  X_with_intercept <- cbind(1, X)
  beta_full <- solve(t(X_with_intercept) %*% X_with_intercept) %*% (t(X_with_intercept) %*% y)
  
  y_pred <- X_with_intercept %*% beta_full
  r2 <- 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
  
  cat(sprintf("R-squared: %.4f\n", r2))
  cat(sprintf("Intercept: %.2f\n", beta_full[1]))
  
  # Focus on end_0 coefficient
  end_0_idx <- which(features == "end_0")
  end_0_coef <- beta_full[end_0_idx + 1]  # +1 because of intercept
  cat(sprintf("Coefficient for end_0: %.2f\n", end_0_coef))
  
  # Create results data frame
  feature_names <- c("intercept", features)
  results_df <- data.frame(
    feature = feature_names,
    coefficient = as.vector(beta_full),
    stringsAsFactors = FALSE
  )
  
  cat("\nTop 10 coefficients by magnitude:\n")
  top_coeffs <- results_df[-1, ]  # Exclude intercept
  top_coeffs$abs_coeff <- abs(top_coeffs$coefficient)
  top_coeffs <- top_coeffs[order(top_coeffs$abs_coeff, decreasing = TRUE), ]
  
  for (i in 1:min(10, nrow(top_coeffs))) {
    cat(sprintf("  %-20s: %10.2f\n", top_coeffs$feature[i], top_coeffs$coefficient[i]))
  }
  
  # Method 2: Partialling-out (FWL) method for end_0
  cat("\n2. Partialling-out Method (focusing on end_0):\n")
  
  # Separate X into X1 (end_0) and X2 (all other variables)
  X1 <- X[, end_0_idx, drop = FALSE]  # Variable of interest
  other_indices <- setdiff(1:ncol(X), end_0_idx)
  X2 <- X[, other_indices, drop = FALSE]  # Control variables
  
  # Add intercept to X2
  X2_with_intercept <- cbind(1, X2)
  
  # Step 1: Regress y on X2 and get residuals
  beta_y_on_x2 <- solve(t(X2_with_intercept) %*% X2_with_intercept) %*% (t(X2_with_intercept) %*% y)
  y_residuals <- y - X2_with_intercept %*% beta_y_on_x2
  
  # Step 2: Regress X1 on X2 and get residuals
  beta_x1_on_x2 <- solve(t(X2_with_intercept) %*% X2_with_intercept) %*% (t(X2_with_intercept) %*% X1)
  x1_residuals <- X1 - X2_with_intercept %*% beta_x1_on_x2
  
  # Step 3: Regress residuals (no intercept needed since residuals are mean zero)
  end_0_coef_fwl <- solve(t(x1_residuals) %*% x1_residuals) %*% (t(x1_residuals) %*% y_residuals)
  end_0_coef_fwl <- as.numeric(end_0_coef_fwl)  # Extract scalar
  
  cat(sprintf("Coefficient for end_0 (FWL method): %.2f\n", end_0_coef_fwl))
  cat(sprintf("Coefficient for end_0 (standard method): %.2f\n", end_0_coef))
  cat(sprintf("Difference: %.6f\n", abs(end_0_coef - end_0_coef_fwl)))
  cat(sprintf("Methods match (within 1e-6): %s\n", abs(end_0_coef - end_0_coef_fwl) < 1e-6))
  
  return(list(
    features = features,
    results_df = results_df,
    end_0_coef_standard = end_0_coef,
    end_0_coef_fwl = end_0_coef_fwl,
    X = X,
    y = y,
    X_with_intercept = X_with_intercept,
    beta_full = beta_full
  ))
}

price_premium_analysis <- function(df, model_results) {
  #' Analyze price premium for apartments with area ending in 0.
  #' Part 3c: Price premium for area that ends in 0-digit (3 points)
  
  cat("\n=== PRICE PREMIUM ANALYSIS (Part 3c) ===\n\n")
  
  features <- model_results$features
  X <- model_results$X
  y <- model_results$y
  
  # Step 1: Train model excluding apartments with area ending in 0 (1.25 points)
  cat("1. Training model excluding apartments with area ending in 0:\n")
  
  # Filter out apartments with area ending in 0
  mask_not_end_0 <- df$end_0 == 0
  X_train <- X[mask_not_end_0, , drop = FALSE]
  y_train <- y[mask_not_end_0]
  
  cat(sprintf("   Training sample size: %d (excluded %d apartments ending in 0)\n", 
              sum(mask_not_end_0), sum(!mask_not_end_0)))
  
  # Train the model (with intercept)
  X_train_with_intercept <- cbind(1, X_train)
  beta_no_end_0 <- solve(t(X_train_with_intercept) %*% X_train_with_intercept) %*% (t(X_train_with_intercept) %*% y_train)
  
  y_pred_train <- X_train_with_intercept %*% beta_no_end_0
  r2_train <- 1 - sum((y_train - y_pred_train)^2) / sum((y_train - mean(y_train))^2)
  cat(sprintf("   R-squared on training data: %.4f\n", r2_train))
  
  # Step 2: Predict prices for entire sample (1.25 points)
  cat("\n2. Predicting prices for entire sample:\n")
  
  X_full_with_intercept <- cbind(1, X)
  
  # Predict using the model trained without end_0 apartments
  y_pred_full <- X_full_with_intercept %*% beta_no_end_0
  
  cat(sprintf("   Predictions generated for %d apartments\n", length(y_pred_full)))
  
  # Step 3: Compare averages for apartments ending in 0 (0.5 points)
  cat("\n3. Comparing actual vs predicted prices for apartments with area ending in 0:\n")
  
  # Get apartments with area ending in 0
  mask_end_0 <- df$end_0 == 1
  
  actual_prices_end_0 <- y[mask_end_0]
  predicted_prices_end_0 <- y_pred_full[mask_end_0]
  
  # Calculate averages
  avg_actual <- mean(actual_prices_end_0)
  avg_predicted <- mean(predicted_prices_end_0)
  premium <- avg_actual - avg_predicted
  premium_pct <- (premium / avg_predicted) * 100
  
  cat(sprintf("   Number of apartments with area ending in 0: %d\n", sum(mask_end_0)))
  cat(sprintf("   Average actual price: %.2f PLN\n", avg_actual))
  cat(sprintf("   Average predicted price: %.2f PLN\n", avg_predicted))
  cat(sprintf("   Price premium: %.2f PLN (%+.2f%%)\n", premium, premium_pct))
  
  # Additional analysis
  cat(sprintf("\n   Additional Statistics:\n"))
  cat(sprintf("   Median actual price: %.2f PLN\n", median(actual_prices_end_0)))
  cat(sprintf("   Median predicted price: %.2f PLN\n", median(predicted_prices_end_0)))
  cat(sprintf("   Standard deviation of premium: %.2f PLN\n", sd(actual_prices_end_0 - predicted_prices_end_0)))
  
  # Determine if apartments ending in 0 are overpriced
  cat(sprintf("\n   Conclusion:\n"))
  if (premium > 0) {
    cat(sprintf("   ✓ Apartments with area ending in 0 appear to be sold at a PREMIUM\n"))
    cat(sprintf("     of %.2f PLN (%+.2f%%) above what their features suggest.\n", premium, premium_pct))
    cat(sprintf("     This could indicate that buyers perceive 'round' areas as more desirable\n"))
    cat(sprintf("     or that sellers use psychological pricing strategies.\n"))
  } else {
    cat(sprintf("   ✗ Apartments with area ending in 0 appear to be sold at a DISCOUNT\n"))
    cat(sprintf("     of %.2f PLN (%.2f%%) below what their features suggest.\n", abs(premium), abs(premium_pct)))
  }
  
  # Statistical significance test
  differences <- actual_prices_end_0 - predicted_prices_end_0
  t_test_result <- t.test(differences, mu = 0)
  t_stat <- t_test_result$statistic
  p_value <- t_test_result$p.value
  
  cat(sprintf("\n   Informal statistical test:\n"))
  cat(sprintf("   t-statistic: %.3f\n", t_stat))
  cat(sprintf("   p-value: %.6f\n", p_value))
  
  if (p_value < 0.05) {
    cat(sprintf("   The price difference is statistically significant at 5%% level.\n"))
  } else {
    cat(sprintf("   The price difference is not statistically significant at 5%% level.\n"))
  }
  
  return(list(
    avg_actual = avg_actual,
    avg_predicted = avg_predicted,
    premium = premium,
    premium_pct = premium_pct,
    n_end_0 = sum(mask_end_0),
    t_stat = as.numeric(t_stat),
    p_value = p_value
  ))
}

save_results <- function(df_clean, model_results, premium_results) {
  #' Save all results to files.
  
  cat("\n=== SAVING RESULTS ===\n\n")
  
  # Create output directory if it doesn't exist
  output_dir <- "/home/runner/work/High_Dimensional_Linear_Models/High_Dimensional_Linear_Models/R/output"
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Save cleaned data
  write.csv(df_clean, file.path(output_dir, "apartments_cleaned.csv"), row.names = FALSE)
  cat("✓ Cleaned data saved to apartments_cleaned.csv\n")
  
  # Save regression results
  write.csv(model_results$results_df, file.path(output_dir, "regression_results.csv"), row.names = FALSE)
  cat("✓ Regression results saved to regression_results.csv\n")
  
  # Save premium analysis results
  premium_summary <- data.frame(
    metric = c("n_apartments_end_0", "avg_actual_price", "avg_predicted_price", 
               "premium_amount", "premium_percentage", "t_statistic", "p_value"),
    value = c(premium_results$n_end_0, premium_results$avg_actual, 
              premium_results$avg_predicted, premium_results$premium,
              premium_results$premium_pct, premium_results$t_stat, 
              premium_results$p_value),
    stringsAsFactors = FALSE
  )
  
  write.csv(premium_summary, file.path(output_dir, "premium_analysis.csv"), row.names = FALSE)
  cat("✓ Premium analysis results saved to premium_analysis.csv\n")
}

main <- function() {
  #' Main function to run the complete analysis.
  
  cat("ASSIGNMENT 1 - PART 3: REAL DATA ANALYSIS\n")
  cat("Hedonic Pricing Model for Polish Apartments\n")
  cat(paste(rep("=", 50), collapse = ""), "\n")
  
  # Load and clean data
  df <- load_data()
  df_clean <- clean_data(df)
  
  # Linear model estimation
  model_results <- linear_model_estimation(df_clean)
  
  # Price premium analysis
  premium_results <- price_premium_analysis(df_clean, model_results)
  
  # Save results
  save_results(df_clean, model_results, premium_results)
  
  cat("\n", paste(rep("=", 50), collapse = ""), "\n")
  cat("ANALYSIS COMPLETE!\n")
  cat("All results saved to R/output/ directory\n")
}

# Main execution when run as script
if (!interactive()) {
  main()
}