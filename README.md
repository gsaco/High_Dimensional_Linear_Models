# High Dimensional Linear Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![R](https://img.shields.io/badge/R-4.0%2B-blue)](https://www.r-project.org/)
[![Julia](https://img.shields.io/badge/Julia-1.6%2B-purple)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains implementations of econometric analyses for high-dimensional linear models in **three programming languages**

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
- **Method**: Employ partitioned matrix algebra and block matrix inversion to demonstrate coefficient equivalence
- **Key Findings**: 
  - $\hat{\beta_1} = (\tilde{X_1}'\tilde{X_1})^{-1}\tilde{X_1}'\tilde{y}$ where residuals are obtained by projecting out control variables
  - $(X_1'M_{X_2}X_1)^{-1}X_1'M_{X_2}y$ yields identical coefficients regardless of estimation sequence
    
### Part 2: Overfitting Analysis (8 points)
- **Objective**: Demonstrate overfitting through polynomial feature expansion
- **Method**: Simulate data with exponential relationship and analyze R² metrics across increasing model complexity
- **Key Findings**: 
  - R² on full sample increases monotonically with features (0.725 → 0.995)

### Part 3: Hedonic Pricing Model (9 points)  
- **Objective**: Investigate pricing effects in Polish real estate market
- **Method**: Analyze 110,191 apartment listings using hedonic regression with area-digit dummies
- **Key Findings**:
  - Apartments with areas ending in "0" command a **1.88% price premium** (16,164 PLN)
  - Premium is **statistically significant** (p < 0.001), indicating systematic pricing behavior

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

## Justification of each step for Part 3 📚

### The Missing Values Challenge: A Methodological Decision

Our analysis began with 110,191 Polish apartment listings but encountered substantial missing value challenges in the dataset.

**Missing Values Found:**
- `buildingmaterial`: 44,265 missing values (40.2% of sample)
- `type`: 23,328 missing values (21.2% of sample) 
- Distance variables: 76-2,931 missing values each
- **Total impact**: 50,874 observations eliminated (46.2% of original sample)

**Methodological Decision:** Rather than pursuing imputation strategies or running regressions with incomplete data, we chose complete case analysis to ensure uniform results across Python, R, and Julia implementations. This approach reduced our analytical sample to **59,317 observations** but provides consistent results across all three programming languages.

---

## Part 3a: Strategic Data Engineering for Behavioral Analysis (2 points)

### **Data Preparation Steps**
Our data engineering focused on testing psychological pricing effects in Polish real estate:

**1. Nonlinear Area Modeling (`area²` creation)**
- Created squared area term to capture potential nonlinear pricing effects
- Allows model to capture both economies and diseconomies of scale in apartment pricing

**2. Binary Feature Standardization (yes/no → 1/0)**
- Converted text-based amenity variables to numeric indicators
- Enables direct interpretation of coefficients as price premiums

**3. Last Digit Dummy Variables (`end_0` through `end_9`)**
- Created indicators for each possible area last digit (0-9)
- `end_9` serves as the reference category
- **Key finding**: `end_0` represents 11.5% of sample (higher than expected 10%)

---

## Part 3b: Econometric Validation Through Multiple Methods (4 points)

### **Model Performance**
Our hedonic pricing model achieved **R² = 0.5933**, explaining 59.33% of price variation across Polish apartments.

**Key Coefficient: end_0 = 25,147.12 PLN**
This represents the estimated premium for apartments with areas ending in 0, controlling for all other apartment characteristics including:
- Area (linear and quadratic terms)
- Distance to various amenities
- Building characteristics (type, material, ownership)
- Apartment features (elevator, balcony, parking, etc.)

### **FWL Theorem Verification**
The Frisch-Waugh-Lovell method yielded exactly **25,147.12 PLN**—identical to the standard regression coefficient, confirming:
- Correct implementation of both methods
- Robustness of the coefficient estimate

---

## Part 3c: Out-of-Sample Premium Analysis (3 points)

### **Experimental Design**
To test whether the premium represents psychological pricing rather than omitted variables:

**Training Phase**: Excluded all 6,848 apartments with areas ending in 0 and estimated hedonic model on remaining 52,469 observations
- Training model R² = 0.5927 (similar to full sample)

**Prediction Phase**: Used trained model to predict prices for apartments with areas ending in 0 based solely on their physical and location characteristics

### **Results**
- **Actual average price** (round-area apartments): 875,919 PLN
- **Predicted average price** (based on features only): 850,595 PLN  
- **Premium**: **25,324 PLN (+2.98%)**

### **Statistical Significance**
- **t-statistic**: 6.005
- **p-value**: 2.03×10⁻⁹
- **Conclusion**: The premium is statistically significant

---

## Cross-Language Implementation: Methodological Consistency

### **Uniform Data Treatment**
To ensure consistent results across programming languages, we implemented identical complete case analysis:

**Python Implementation:** pandas `.notna().all(axis=1)` for complete cases
**R Implementation:** `complete.cases()` function for missing value removal  
**Julia Implementation:** `dropmissing()` for data cleaning

### **Validation Results**
All three implementations produced identical results:
- **Sample size**: 59,317 observations
- **end_0 coefficient**: 25,147.12 PLN  
- **Premium estimate**: 25,324 PLN (+2.98%)
- **Statistical significance**: p < 2×10⁻⁹

This consistency across Python, R, and Julia validates our findings and demonstrates the robustness of the results.

**✅ Analysis Complete: Evidence of pricing premium for apartments with round-numbered areas, validated across


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request for improvements to the analysis or code implementations.

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact the repository maintainer.

---

*This repository demonstrates practical applications of econometric methods and serves as a comprehensive resource for understanding high-dimensional linear models in empirical research.*
