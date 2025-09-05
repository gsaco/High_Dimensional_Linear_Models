# High-Dimensional Linear Models: Overfitting Simulation

This repository contains implementations demonstrating the overfitting phenomenon in high-dimensional linear models across three programming languages: Python, Julia, and R.

## Overview

The simulation demonstrates how increasing the number of polynomial features in a linear regression model affects:
- Training R-squared (always increases)
- Adjusted R-squared (increases then decreases)  
- Out-of-sample R-squared (increases then decreases, showing overfitting)

## Data Generating Process

We use the following nonlinear data generating process:
- f(X) = exp(4 * X) - 1
- Y = f(X) + ε, where ε ~ N(0, σ²)
- n = 1000 observations
- X ~ Uniform(-0.5, 0.5)
- Intercept parameter = 0

## Implementation Files

### Jupyter Notebooks
- `simulation_python.ipynb` - Python implementation using scikit-learn
- `simulation_julia.ipynb` - Julia implementation with native linear algebra
- `simulation_r.ipynb` - R implementation using base R and ggplot2
- `simulation.ipynb` - Main simulation file (Python version)

### Features Tested
The simulation tests polynomial regression with the following numbers of features:
1, 2, 5, 10, 20, 50, 100, 200, 500, 1000

### Metrics Calculated
1. **R-squared (Training)**: Goodness of fit on training data
2. **Adjusted R-squared**: R-squared penalized for number of parameters
3. **Out-of-sample R-squared**: Performance on held-out test data (25% of data)

## Key Results

The simulation demonstrates the classic overfitting pattern:
- Training R² monotonically increases with model complexity
- Adjusted R² initially increases then decreases as complexity penalty dominates
- Out-of-sample R² peaks at moderate complexity then decreases due to overfitting

## Theoretical Background

This demonstrates the **bias-variance tradeoff**:
- **Simple models**: High bias (underfitting), low variance
- **Complex models**: Low bias, high variance (overfitting)
- **Optimal complexity**: Minimizes total error = bias² + variance + noise

## Usage

### Python
```bash
jupyter notebook simulation_python.ipynb
```

### Julia
```bash
jupyter notebook simulation_julia.ipynb
```

### R
```bash
jupyter notebook simulation_r.ipynb
```

## Dependencies

### Python
- numpy
- pandas
- matplotlib
- scikit-learn
- jupyter

### Julia
- Random, Distributions, LinearAlgebra
- DataFrames, CSV
- Plots, StatsPlots
- GLM, StatsBase, MLBase

### R
- ggplot2
- dplyr
- tidyr
- gridExtra

## Educational Objectives

This simulation illustrates fundamental concepts in:
- High-dimensional statistics
- Model selection and validation
- Overfitting and generalization
- Causal inference and machine learning

## License

MIT License - see LICENSE file for details.
