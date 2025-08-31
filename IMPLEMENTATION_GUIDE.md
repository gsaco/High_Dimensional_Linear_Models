# Multi-Language Implementation Guide

This repository now contains complete implementations of the High Dimensional Linear Models assignment in three programming languages: Python, Julia, and R.

## Repository Structure

```
High_Dimensional_Linear_Models/
├── Python/
│   ├── scripts/
│   │   ├── part1_fwl_theorem.py
│   │   ├── part2_overfitting.py
│   │   └── part3_hedonic_pricing.py
│   └── output/          # Generated results and plots
├── Julia/
│   ├── scripts/
│   │   ├── part1_fwl_theorem.jl
│   │   ├── part2_overfitting.jl
│   │   └── part3_hedonic_pricing.jl
│   ├── assignment1_complete.jl
│   └── output/          # Generated results and plots
├── R/
│   ├── scripts/
│   │   ├── part1_fwl_theorem.R
│   │   ├── part2_overfitting.R
│   │   └── part3_hedonic_pricing.R
│   ├── assignment1_complete.R
│   └── output/          # Generated results and plots
├── assignment1_complete.py    # Python master script
└── requirements.txt          # Python dependencies
```

## Quick Start

### Python Implementation
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete analysis
python assignment1_complete.py

# Or run individual parts
python Python/scripts/part1_fwl_theorem.py
python Python/scripts/part2_overfitting.py
python Python/scripts/part3_hedonic_pricing.py
```

### Julia Implementation
```julia
# Install required packages
using Pkg
Pkg.add(["DataFrames", "CSV", "Plots", "Statistics", "StatsBase", "HypothesisTests"])

# Run complete analysis
julia Julia/assignment1_complete.jl

# Or run individual parts
julia Julia/scripts/part1_fwl_theorem.jl
julia Julia/scripts/part2_overfitting.jl
julia Julia/scripts/part3_hedonic_pricing.jl
```

### R Implementation
```r
# Install required packages
install.packages(c("ggplot2", "dplyr", "MASS"))

# Run complete analysis
Rscript R/assignment1_complete.R

# Or run individual parts
Rscript R/scripts/part1_fwl_theorem.R
Rscript R/scripts/part2_overfitting.R
Rscript R/scripts/part3_hedonic_pricing.R
```

## Assignment Parts

### Part 1: Frisch-Waugh-Lovell (FWL) Theorem (3 points)
- **Mathematical proof**: Complete theoretical derivation
- **Numerical verification**: Simulation demonstrating theorem validity
- **Key libraries**: 
  - Python: `numpy`, `sklearn`
  - Julia: `LinearAlgebra`
  - R: Base R matrix operations, `MASS`

### Part 2: Overfitting Analysis (8 points)
- **Data generation**: 1000 observations with polynomial features
- **Analysis**: R², Adjusted R², Out-of-sample performance
- **Visualization**: Three plots showing bias-variance tradeoff
- **Key libraries**:
  - Python: `matplotlib`, `seaborn`, `sklearn`
  - Julia: `Plots.jl`, `DataFrames.jl`
  - R: `ggplot2`, `dplyr`

### Part 3: Hedonic Pricing Model (9 points)
- **Data cleaning**: Feature engineering and dummy variables
- **Linear modeling**: OLS regression with FWL verification
- **Premium analysis**: Psychological pricing effects detection
- **Key libraries**:
  - Python: `pandas`, `sklearn`, `scipy`
  - Julia: `DataFrames.jl`, `HypothesisTests.jl`
  - R: `dplyr`, base R statistical functions

## Output Files

Each implementation generates the same set of output files in their respective directories:

### Generated Plots
- `r2_full_sample.png` - In-sample R² vs features
- `adj_r2_full_sample.png` - Adjusted R² vs features  
- `r2_out_of_sample.png` - Out-of-sample R² vs features

### Generated Data
- `overfitting_results.csv` - Complete overfitting analysis results
- `apartments_cleaned.csv` - Cleaned apartment dataset
- `regression_results.csv` - Full regression coefficients
- `premium_analysis.csv` - Price premium analysis summary

## Key Features

### Equivalent Results
All three implementations produce mathematically equivalent results using the same:
- Random seeds for reproducibility
- Data generation processes
- Statistical methods and algorithms

### Language-Specific Best Practices
- **Python**: Object-oriented design, pandas workflows, sklearn integration
- **Julia**: Multiple dispatch, type stability, performance optimization
- **R**: Functional programming, data.frame operations, statistical modeling

### Educational Value
- Demonstrates same concepts across programming paradigms
- Shows idiomatic usage of each language's ecosystem
- Provides comparative implementation study

## Technical Notes

### Dependencies
- **Python**: Requires numpy, pandas, scikit-learn, matplotlib, seaborn, scipy
- **Julia**: Requires DataFrames, CSV, Plots, Statistics, StatsBase, HypothesisTests
- **R**: Requires ggplot2, dplyr, MASS (base R sufficient for most functionality)

### Performance Considerations
- Julia implementation optimized for numerical performance
- Python implementation optimized for readability and ecosystem integration
- R implementation optimized for statistical analysis workflows

### Reproducibility
All implementations use fixed random seeds (42) to ensure identical results across runs and languages for comparative analysis.

## Contributing

When making changes:
1. Maintain consistency across all three language implementations
2. Ensure mathematical equivalence of results
3. Follow language-specific style guidelines
4. Update documentation to reflect changes in all versions

## Usage in Teaching

This multi-language implementation is ideal for:
- Econometrics courses comparing programming languages
- Demonstrating algorithm implementation across paradigms
- Student choice of preferred programming environment
- Comparative analysis of statistical computing approaches