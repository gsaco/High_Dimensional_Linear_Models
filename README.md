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
├── Python/           # Python implementation
│   ├── input/       # Data files
│   ├── output/      # Results and visualizations  
│   └── scripts/     # Jupyter notebooks
├── R/               # R implementation
│   ├── input/       # Data files
│   ├── output/      # Results and visualizations
│   └── scripts/     # R notebooks
├── Julia/           # Julia implementation
│   ├── input/       # Data files
│   ├── output/      # Results and visualizations
│   └── scripts/     # Julia notebooks
├── requirements.txt # Python dependencies
└── README.md       # This file
```

## 🎯 Analytical Components

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
pip install -r requirements.txt

# Run analyses
cd Python/scripts/
jupyter notebook part2_overfitting.ipynb
jupyter notebook part3_hedonic_pricing.ipynb
```

#### R
```bash
# Install required packages (run in R console)
install.packages(c("dplyr", "ggplot2", "readr", "broom"))

# Run analyses  
cd R/scripts/
# Open .ipynb files in Jupyter with R kernel or RStudio
```

#### Julia
```bash
# Install packages (run in Julia REPL)
using Pkg
Pkg.add(["DataFrames", "CSV", "GLM", "Plots", "StatsPlots"])

# Run analyses
cd Julia/scripts/  
# Open .ipynb files in Jupyter with Julia kernel
```

## 📈 Key Results & Insights

### Overfitting Analysis Results
| Features | R² (Full) | Adj R² (Full) | R² (Out-of-Sample) |
|----------|-----------|---------------|-------------------|
| 1        | 0.725     | 0.725         | 0.716            |
| 2        | 0.964     | 0.963         | 0.966            |
| 5        | 0.995     | 0.995         | 0.995            |
| 50       | 0.995     | 0.995         | 0.995            |
| 1000     | 0.995     | NaN           | 0.995            |

**Interpretation**: The analysis reveals classical overfitting patterns where standard R² misleadingly suggests improvement with complexity, while adjusted R² and out-of-sample performance provide more reliable model selection guidance.

### Hedonic Pricing Results
- **Dataset**: 110,191 Polish apartment listings
- **Price Premium for "Round" Areas**: +16,164 PLN (+1.88%)
- **Statistical Significance**: t = 5.87, p < 0.001
- **Economic Interpretation**: Evidence of psychological pricing effects in real estate markets

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
