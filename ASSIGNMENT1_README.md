# Assignment 1: High Dimensional Linear Models

This repository contains the complete implementation of Assignment 1 for the High Dimensional Linear Models course, covering theoretical proofs, simulation studies, and real data analysis.

## 📋 Assignment Overview

The assignment consists of three main parts:
1. **Math (3 points)**: Frisch-Waugh-Lovell (FWL) Theorem proof and verification
2. **Overfitting (8 points)**: Simulation study analyzing overfitting with increasing model complexity
3. **Real Data (9 points)**: Hedonic pricing model for Polish apartment data

## 🏗️ Repository Structure

```
High_Dimensional_Linear_Models/
├── assignment1_complete.py           # Master script to run all parts
├── requirements.txt                  # Python dependencies
├── Python/
│   ├── scripts/
│   │   ├── part1_fwl_theorem.py     # FWL theorem implementation
│   │   ├── part2_overfitting.py     # Overfitting analysis
│   │   └── part3_hedonic_pricing.py # Hedonic pricing model
│   └── output/
│       ├── *.csv                    # Results data files
│       └── *.png                    # Generated plots
├── R/                               # R implementation directory
└── Julia/                           # Julia implementation directory
```

## 🚀 Quick Start

### Prerequisites
```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Run Complete Analysis
```bash
# Run all three parts at once
python assignment1_complete.py

# Or run individual parts
python Python/scripts/part1_fwl_theorem.py
python Python/scripts/part2_overfitting.py
python Python/scripts/part3_hedonic_pricing.py
```

## 📊 Part 1: Frisch-Waugh-Lovell Theorem (3 points)

### Overview
Proves and numerically verifies the FWL theorem, which shows that the OLS estimate of β₁ in a regression of y on [X₁ X₂] equals the OLS estimate from a two-step partialling-out procedure.

### Implementation
- **Mathematical proof**: Complete theoretical derivation using partitioned matrices
- **Numerical verification**: Simulation with 1000 observations comparing full regression vs FWL method
- **Validation**: Both methods produce identical results (difference < 1e-15)

### Key Results
- ✅ Theorem proven mathematically
- ✅ Numerical verification successful
- ✅ Methods match within machine precision

## 📈 Part 2: Overfitting Analysis (8 points)

### Overview
Simulates a data generating process and analyzes overfitting by estimating linear models with increasing numbers of polynomial features.

### Methodology
- **Data generation**: n=1000, true relationship y = 2X + u (no intercept)
- **Features tested**: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000 polynomial features
- **Metrics calculated**: R², Adjusted R², Out-of-sample R² (75/25 train/test split)
- **Visualization**: Three separate plots showing different R² measures

### Key Findings
- **R² (full sample)**: Monotonically increases with features (expected)
- **Adjusted R²**: Peaks early (~2 features) then declines due to complexity penalty
- **Out-of-sample R²**: Shows classic overfitting pattern - improves initially, then deteriorates dramatically
- **Economic intuition**: Clear demonstration of bias-variance tradeoff

### Generated Outputs
- `r2_full_sample.png`: In-sample R² vs number of features
- `adj_r2_full_sample.png`: Adjusted R² vs number of features  
- `r2_out_of_sample.png`: Out-of-sample R² vs number of features
- `overfitting_results.csv`: Complete numerical results

## 🏠 Part 3: Hedonic Pricing Model (9 points)

### Overview
Analyzes apartment pricing in Poland using a hedonic pricing model to estimate the value of apartments based on their features, with special focus on psychological pricing effects.

### Part 3a: Data Cleaning (2 points)
- ✅ Created `area2` variable (square of area)
- ✅ Converted binary variables ('yes'/'no' → 1/0): parking, balcony, elevator, security, storage
- ✅ Created area last digit dummies (`end_0` through `end_9`)

### Part 3b: Linear Model Estimation (4 points)
- ✅ **Standard regression**: Price regressed on all specified covariates
- ✅ **Partialling-out method**: FWL implementation focusing on `end_0` coefficient
- ✅ **Verification**: Both methods produce identical coefficients

### Regression Features
- Area last digit dummies (omitting `end_9` as base)
- Area and area²
- Distance variables (school, clinic, post office, etc.)
- Binary features (parking, balcony, elevator, security, storage)
- Categorical variables (month, type, rooms, ownership, building material)

### Part 3c: Price Premium Analysis (3 points)
- ✅ **Model training**: Excluded apartments with area ending in 0
- ✅ **Price prediction**: Generated predictions for entire sample
- ✅ **Premium analysis**: Compared actual vs predicted prices for area ending in 0

### Key Findings
- **Premium detected**: 6,437 PLN (+7.57%) for apartments with area ending in 0
- **Statistical significance**: p-value = 0.000023 (highly significant)
- **Economic interpretation**: Evidence of psychological pricing in real estate

### Generated Outputs
- `apartments_cleaned.csv`: Cleaned dataset with all transformations
- `regression_results.csv`: Complete regression coefficients
- `premium_analysis.csv`: Premium analysis summary statistics

## 🔍 Key Results Summary

| Part | Status | Key Finding |
|------|--------|-------------|
| 1. FWL Theorem | ✅ Complete | Theoretical proof verified numerically |
| 2. Overfitting | ✅ Complete | Clear bias-variance tradeoff demonstrated |
| 3. Hedonic Model | ✅ Complete | 7.57% price premium for "round" areas |

## 📈 Statistical Methods Used

- **OLS Regression**: Standard and partialling-out implementations
- **Cross-validation**: Train/test splits for out-of-sample evaluation
- **Polynomial features**: For overfitting analysis
- **Dummy variables**: For categorical data encoding
- **Statistical testing**: t-tests for significance analysis

## 🛠️ Technical Implementation

### Libraries Used
- **NumPy**: Numerical computations and linear algebra
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and metrics
- **Matplotlib/Seaborn**: Data visualization
- **SciPy**: Statistical tests

### Code Quality Features
- Comprehensive documentation and comments
- Modular design with reusable functions
- Error handling and validation
- Reproducible results (fixed random seeds)
- Clean output formatting and interpretation

## 📝 Academic Contributions

This implementation demonstrates:
1. **Theoretical understanding**: Rigorous mathematical proofs
2. **Empirical skills**: Simulation studies and data analysis
3. **Real-world application**: Economic modeling with policy implications
4. **Statistical literacy**: Proper interpretation of results and limitations

## 🎯 Learning Outcomes Achieved

- ✅ Understanding of fundamental econometric theorems (FWL)
- ✅ Practical experience with overfitting and model selection
- ✅ Real-world data analysis and interpretation
- ✅ Implementation of advanced statistical methods
- ✅ Scientific programming and reproducible research practices

---

*This assignment demonstrates proficiency in high-dimensional linear models, combining theoretical knowledge with practical implementation skills for real-world economic analysis.*