# High Dimensional Linear Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0%2B-blue)](https://www.r-project.org/)
[![Julia](https://img.shields.io/badge/Julia-1.6%2B-purple)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains comprehensive implementations of advanced econometric analyses for high-dimensional linear models in **three programming languages**. The project demonstrates key concepts in causal inference, machine learning, and econometric modeling through practical applications.

## 📊 Project Structure

The repository is organized by programming language, with each implementation providing identical analytical results:

```
High_Dimensional_Linear_Models/
├── Python/             # Python implementation
│   ├── input/         # Data files
│   ├── output/        # Results, CSV files, and PNG plots  
│   ├── scripts/       # Jupyter notebooks
│   └── requirements.txt  # Python dependencies
├── R/                 # R implementation
│   ├── input/         # Data files
│   ├── output/        # Results, CSV files, and PNG plots
│   ├── scripts/       # R notebooks
│   └── requirements.txt  # R dependencies
├── Julia/             # Julia implementation
│   ├── input/         # Data files
│   ├── output/        # Results, CSV files, and PNG plots
│   ├── scripts/       # Julia notebooks
│   └── requirements.txt  # Julia dependencies
└── README.md         # This file
```

## 🎯 Analytical Components

### Part 1: Frisch-Waugh-Lovell Theorem (3 points)
- **Objective**: Provide rigorous mathematical proof of the equivalence between full regression and partialling-out procedures
- **Method**: Employ partitioned matrix algebra and block matrix inversion to demonstrate coefficient equivalence across estimation approaches
- **Key Findings**: 
  - **Mathematical equivalence established**: $\hat{\beta_1} = (\tilde{X_1}'\tilde{X_1})^{-1}\tilde{X_1}'\tilde{y}$ where residuals are obtained by projecting out control variables
  - **Fundamental identity proven**: $(X_1'M_{X_2}X_1)^{-1}X_1'M_{X_2}y$ yields identical coefficients regardless of estimation sequence
  - **Theoretical foundation demonstrated** for computational efficiency and causal identification strategies in high-dimensional settings
    
### Part 2: Overfitting Analysis (8 points)
- **Objective**: Demonstrate bias-variance tradeoff through polynomial feature expansion
- **Method**: Simulate data with exponential relationship and analyze R² metrics across increasing model complexity
- **Key Findings**: 
  - R² on full sample increases monotonically with features (0.725 → 0.995)
  - Adjusted R² peaks at moderate complexity then declines due to overfitting
  - Out-of-sample R² stabilizes around 0.995, showing optimal model complexity

### Part 3: Hedonic Pricing Model (9 points)  
- **Objective**: Investigate psychological pricing effects in Polish real estate market
- **Method**: Analyze 110,191 apartment listings using hedonic regression with area-digit dummies
- **Key Findings**:
  - Apartments with areas ending in "0" command a **1.88% price premium** (16,164 PLN)
  - Premium is **statistically significant** (p < 0.001), indicating systematic pricing behavior
  - Results suggest psychological anchoring effects in real estate valuation

## 📊 Outputs & Visualizations

When you run the scripts, all results are automatically saved to each language's `output/` directory:

### Generated Files
- **CSV Files**: Statistical results, cleaned data, regression outputs
- **PNG Plots**: High-resolution visualizations (300 DPI)
  - `r2_full_sample.png` - R-squared on full sample vs features
  - `adj_r2_full_sample.png` - Adjusted R-squared vs features  
  - `r2_out_of_sample.png` - Out-of-sample R-squared vs features
  - `hedonic_pricing_analysis.png` - Real estate pricing analysis plots

### Language-Specific Dependencies
Each language folder contains its own `requirements.txt`:
- **Python/requirements.txt**: NumPy, Pandas, Matplotlib, Scikit-learn, etc.
- **R/requirements.txt**: dplyr, ggplot2, MASS, readr, etc.
- **Julia/requirements.txt**: DataFrames, Plots, GLM, StatsPlots, etc.

## 🚀 Quick Start

### Prerequisites
Ensure you have one of the following installed:
- **Python 3.8+** with pip
- **R 4.0+** with required packages  
- **Julia 1.6+** with package manager

### Installation & Usage

#### Python
```bash
# Install dependencies
pip install -r Python/requirements.txt

# Run analyses (plots are automatically saved to Python/output/)
cd Python/scripts/
jupyter notebook part2_overfitting.ipynb
jupyter notebook part3_hedonic_pricing.ipynb
```

#### R
```bash
# Install required packages (see R/requirements.txt for full list)
R -e "install.packages(c('dplyr', 'ggplot2', 'MASS', 'readr', 'broom'))"

# Run analyses (plots are automatically saved to R/output/)
cd R/scripts/
# Open .ipynb files in Jupyter with R kernel or RStudio
```

#### Julia
```bash
# Install packages (see Julia/requirements.txt for full list)
julia -e "using Pkg; Pkg.add([\"DataFrames\", \"CSV\", \"GLM\", \"Plots\", \"StatsPlots\"])"

# Run analyses (plots are automatically saved to Julia/output/)
cd Julia/scripts/  
# Open .ipynb files in Jupyter with Julia kernel
```
## 🛠️ Technical Implementation

### Dependencies
- **Python**: NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, SciPy, Jupyter
- **R**: dplyr, ggplot2, readr, broom, knitr
- **Julia**: DataFrames, CSV, GLM, Plots, StatsPlots, IJulia

### Data Sources
- **Overfitting Analysis**: Simulated exponential data (n=1,000)
- **Hedonic Pricing**: Real Polish apartment data (110,191 observations)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request for improvements to the analysis or code implementations.

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact the repository maintainer.

---

*This repository demonstrates practical applications of econometric methods and serves as a comprehensive resource for understanding high-dimensional linear models in empirical research.*
