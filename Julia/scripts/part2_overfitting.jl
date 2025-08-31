"""
Assignment 1 - Part 2: Overfitting Analysis
Overfitting (8 points)

This module simulates a data generating process and analyzes overfitting
by estimating linear models with increasing numbers of features.

Author: Julia implementation for gsaco/High_Dimensional_Linear_Models
"""

using LinearAlgebra
using Random
using Printf
using Plots
using DataFrames
using CSV

# Set plotting backend
gr()

function generate_data(n=1000; seed=42)
    """
    Generate data following the specification in Lab2 with only 2 variables X and Y.
    Intercept parameter is set to zero as requested.
    
    Parameters:
    -----------
    n : Int
        Sample size (default: 1000)
    seed : Int
        Random seed for reproducibility
        
    Returns:
    --------
    X : Matrix
        Feature matrix
    y : Vector
        Target variable
    """
    Random.seed!(seed)
    
    # Generate X (single feature initially)
    X = randn(n, 1)
    
    # Generate error term
    u = randn(n)
    
    # Generate y with no intercept (as requested)
    # True relationship: y = 2*X + u
    beta_true = 2.0
    y = beta_true * X[:, 1] + u
    
    return X, y
end

function create_polynomial_features(X, n_features)
    """
    Create polynomial features up to n_features.
    
    Parameters:
    -----------
    X : Matrix
        Original feature matrix (n x 1)
    n_features : Int
        Number of features to create
        
    Returns:
    --------
    X_poly : Matrix
        Extended feature matrix with polynomial features
    """
    n_samples = size(X, 1)
    X_poly = zeros(n_samples, n_features)
    
    for i in 1:n_features
        X_poly[:, i] = X[:, 1] .^ i  # x^1, x^2, x^3, etc.
    end
    
    return X_poly
end

function calculate_adjusted_r2(r2, n, k)
    """
    Calculate adjusted R-squared.
    
    Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - k - 1)]
    
    Parameters:
    -----------
    r2 : Float64
        R-squared value
    n : Int
        Sample size
    k : Int
        Number of features (excluding intercept)
        
    Returns:
    --------
    adj_r2 : Float64
        Adjusted R-squared
    """
    if n - k - 1 <= 0
        return NaN
    end
    
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    return adj_r2
end

function r2_score(y_true, y_pred)
    """Calculate R-squared score."""
    ss_res = sum((y_true - y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    return 1 - (ss_res / ss_tot)
end

function train_test_split(X, y; test_size=0.25, random_state=42)
    """Split data into training and testing sets."""
    Random.seed!(random_state)
    n = length(y)
    n_test = round(Int, n * test_size)
    indices = randperm(n)
    
    test_indices = indices[1:n_test]
    train_indices = indices[n_test+1:end]
    
    return X[train_indices, :], X[test_indices, :], y[train_indices], y[test_indices]
end

function overfitting_analysis()
    """
    Main function to perform overfitting analysis.
    """
    println("=== OVERFITTING ANALYSIS ===\n")
    
    # Generate data
    X, y = generate_data(1000, seed=42)
    
    @printf("Generated data with n=%d observations\n", length(y))
    println("True relationship: y = 2*X + u")
    @printf("X shape: (%d, %d)\n", size(X)...)
    @printf("y shape: (%d,)\n", length(y))
    println()
    
    # Number of features to test
    n_features_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    # Storage for results
    results = DataFrame(
        n_features = Int[],
        r2_full = Float64[],
        adj_r2_full = Float64[],
        r2_out_of_sample = Float64[]
    )
    
    println("Analyzing overfitting for different numbers of features...")
    println("Features | R² (full) | Adj R² (full) | R² (out-of-sample)")
    println("-" * 60)
    
    for n_feat in n_features_list
        try
            # Create polynomial features
            X_poly = create_polynomial_features(X, n_feat)
            
            # Split data into train/test (75%/25%)
            X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.25, random_state=42)
            
            # Fit model on full sample (no intercept as requested)
            beta_full = (X_poly' * X_poly) \ (X_poly' * y)
            y_pred_full = X_poly * beta_full
            r2_full = r2_score(y, y_pred_full)
            
            # Calculate adjusted R²
            adj_r2_full = calculate_adjusted_r2(r2_full, length(y), n_feat)
            
            # Fit model on training data and predict on test data
            beta_train = (X_train' * X_train) \ (X_train' * y_train)
            y_pred_test = X_test * beta_train
            r2_out_of_sample = r2_score(y_test, y_pred_test)
            
            # Store results
            push!(results, (n_feat, r2_full, adj_r2_full, r2_out_of_sample))
            
            @printf("%8d | %9.4f | %12.4f | %17.4f\n", n_feat, r2_full, adj_r2_full, r2_out_of_sample)
            
        catch e
            println("Error with $n_feat features: $e")
            # Still append to maintain list length
            push!(results, (n_feat, NaN, NaN, NaN))
        end
    end
    
    println()
    return results
end

function create_plots(df_results)
    """
    Create three separate plots for R-squared analysis.
    
    Parameters:
    -----------
    df_results : DataFrame
        Results from overfitting analysis
    """
    println("Creating plots...")
    
    # Create output directory if it doesn't exist
    output_dir = "/home/runner/work/High_Dimensional_Linear_Models/High_Dimensional_Linear_Models/Julia/output"
    mkpath(output_dir)
    
    # Plot 1: R-squared (full sample)
    p1 = plot(df_results.n_features, df_results.r2_full,
              marker=:circle, linewidth=2, markersize=6, color=:blue,
              title="R-squared on Full Sample vs Number of Features",
              xlabel="Number of Features", ylabel="R-squared",
              xscale=:log10, ylims=(0, 1), grid=true,
              titlefontsize=12, labelfontsize=10)
    
    savefig(p1, joinpath(output_dir, "r2_full_sample.png"))
    
    # Plot 2: Adjusted R-squared (full sample)  
    p2 = plot(df_results.n_features, df_results.adj_r2_full,
              marker=:square, linewidth=2, markersize=6, color=:green,
              title="Adjusted R-squared on Full Sample vs Number of Features",
              xlabel="Number of Features", ylabel="Adjusted R-squared",
              xscale=:log10, grid=true,
              titlefontsize=12, labelfontsize=10)
    
    savefig(p2, joinpath(output_dir, "adj_r2_full_sample.png"))
    
    # Plot 3: Out-of-sample R-squared
    p3 = plot(df_results.n_features, df_results.r2_out_of_sample,
              marker=:utriangle, linewidth=2, markersize=6, color=:red,
              title="Out-of-Sample R-squared vs Number of Features",
              xlabel="Number of Features", ylabel="Out-of-Sample R-squared",
              xscale=:log10, grid=true,
              titlefontsize=12, labelfontsize=10)
    
    savefig(p3, joinpath(output_dir, "r2_out_of_sample.png"))
    
    println("Plots saved to Julia/output/ directory")
    
    return p1, p2, p3
end

function interpret_results(df_results)
    """
    Provide interpretation and intuition for the results.
    
    Parameters:
    -----------
    df_results : DataFrame
        Results from overfitting analysis
    """
    println("\n=== RESULTS INTERPRETATION ===\n")
    
    println("Key Observations:")
    println("================")
    
    # R-squared observations
    max_r2_full = maximum(df_results.r2_full[.!isnan.(df_results.r2_full)])
    max_r2_idx = findfirst(x -> x == max_r2_full, df_results.r2_full)
    max_r2_features = df_results.n_features[max_r2_idx]
    
    @printf("1. R-squared (Full Sample):\n")
    @printf("   - Starts at %.4f with 1 feature\n", df_results.r2_full[1])
    @printf("   - Reaches maximum of %.4f with %d features\n", max_r2_full, max_r2_features)
    @printf("   - Shows monotonic increase as expected in in-sample fit\n")
    println()
    
    # Adjusted R-squared observations
    valid_adj_r2 = df_results.adj_r2_full[.!isnan.(df_results.adj_r2_full)]
    if !isempty(valid_adj_r2)
        max_adj_r2 = maximum(valid_adj_r2)
        max_adj_r2_idx = findfirst(x -> x == max_adj_r2, df_results.adj_r2_full)
        max_adj_r2_features = df_results.n_features[max_adj_r2_idx]
        
        @printf("2. Adjusted R-squared (Full Sample):\n")
        @printf("   - Peaks at %.4f with %d features\n", max_adj_r2, max_adj_r2_features)
        @printf("   - Then declines as the penalty for additional features outweighs benefit\n")
        @printf("   - Becomes negative when model is severely overfitted\n")
        println()
    end
    
    # Out-of-sample observations
    valid_oos_r2 = df_results.r2_out_of_sample[.!isnan.(df_results.r2_out_of_sample)]
    if !isempty(valid_oos_r2)
        max_oos_r2 = maximum(valid_oos_r2)
        max_oos_r2_idx = findfirst(x -> x == max_oos_r2, df_results.r2_out_of_sample)
        max_oos_r2_features = df_results.n_features[max_oos_r2_idx]
        min_oos_r2 = minimum(valid_oos_r2)
        
        @printf("3. Out-of-Sample R-squared:\n")
        @printf("   - Peaks at %.4f with %d features\n", max_oos_r2, max_oos_r2_features)
        @printf("   - Drops dramatically to %.4f as overfitting increases\n", min_oos_r2)
        @printf("   - Can become negative when predictions are worse than using the mean\n")
        println()
    end
    
    println("Economic Intuition:")
    println("==================")
    println()
    println("1. **Bias-Variance Tradeoff**: As we add more features (higher-order polynomials),")
    println("   we reduce bias but increase variance. Initially, bias reduction dominates,")
    println("   improving out-of-sample performance. Eventually, variance dominates.")
    println()
    println("2. **In-Sample vs Out-of-Sample**: In-sample R² always increases with more features")
    println("   because the model can always fit the training data better. However, this")
    println("   doesn't translate to better prediction on new data.")
    println()
    println("3. **Adjusted R-squared as a Model Selection Tool**: Adjusted R² penalizes model")
    println("   complexity and provides a better guide for model selection than raw R².")
    println()
    println("4. **The Curse of Dimensionality**: With 1000 observations and up to 1000 features,")
    println("   we approach the case where we have as many parameters as observations,")
    println("   leading to perfect in-sample fit but terrible out-of-sample performance.")
    println()
    println("5. **Practical Implications**: This demonstrates why regularization techniques")
    println("   (Ridge, Lasso, Elastic Net) are crucial in high-dimensional settings to")
    println("   prevent overfitting and improve generalization.")
end

# Main execution when run as script
if abspath(PROGRAM_FILE) == @__FILE__
    # Run overfitting analysis
    results_df = overfitting_analysis()
    
    # Create plots
    create_plots(results_df)
    
    # Interpret results
    interpret_results(results_df)
    
    # Save results to CSV
    output_dir = "/home/runner/work/High_Dimensional_Linear_Models/High_Dimensional_Linear_Models/Julia/output"
    mkpath(output_dir)
    CSV.write(joinpath(output_dir, "overfitting_results.csv"), results_df)
    println("\nResults saved to Julia/output/overfitting_results.csv")
end