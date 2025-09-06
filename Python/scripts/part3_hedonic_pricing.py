#!/usr/bin/env python
# coding: utf-8

# # Assignment 1 - Part 3: Real Data Analysis - Hedonic Pricing Model
# ## Real data (9 points)
# 
# This notebook implements hedonic pricing model analysis using real apartment data from Poland. We will analyze whether apartments with areas ending in "0" (round numbers) command a price premium, which could indicate psychological pricing effects in the real estate market.
# 
# ## Analysis Structure:
# - **Part 3a (2 points)**: Data cleaning and feature engineering
# - **Part 3b (4 points)**: Linear model estimation using both standard and partialling-out methods
# - **Part 3c (3 points)**: Price premium analysis for "round" areas

# ## Import Required Libraries

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


# ## Data Loading
# 
# Let's load the real apartment data from the repository.

# In[27]:


def load_data():
    """
    Load apartment data from the repository.
    """
    print("Loading apartment data from repository...")

    # Load the real apartments.csv file from the input folder
    data_path = '../input/apartments.csv'  # Relative path from scripts/ to input
    df = pd.read_csv(data_path)

    print(f"Loaded data with {len(df)} observations and {len(df.columns)} variables")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")

    # Display basic info about the dataset
    print("\nDataset Info:")
    print(df.info())

    return df

# Load the data
df = load_data()


# ## Data Exploration
# 
# Let's explore the dataset to understand its structure and characteristics.

# In[28]:


# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])


# ## Part 3a: Data Cleaning (2 points)
# 
# We need to perform the following data cleaning tasks:
# 1. Create `area2` variable (square of area)
# 2. Convert binary variables ('yes'/'no' → 1/0)
# 3. Create area last digit dummies (`end_0` through `end_9`)

# In[29]:


def clean_data(df):
    """
    Perform data cleaning as specified in Part 3a.

    Tasks:
    1. Create area2 variable (square of area)
    2. Convert binary variables to dummy variables (yes/no -> 1/0)
    3. Create last digit dummy variables for area (end_0 to end_9)
    """
    print("\n=== DATA CLEANING (Part 3a) ===\n")

    df_clean = df.copy()

    # 1. Create area2 variable (0.25 points)
    df_clean['area2'] = df_clean['area'] ** 2
    print("✓ Created area2 variable (square of area)")

    # 2. Convert binary variables to dummy variables (0.75 points)
    # First, let's identify the binary variables in our dataset
    binary_vars = []
    for col in df_clean.columns:
        if col.startswith('has') and df_clean[col].dtype == 'object':
            binary_vars.append(col)

    print(f"\nIdentified binary variables: {binary_vars}")

    for var in binary_vars:
        # Convert 'yes'/'no' to 1/0
        df_clean[var] = (df_clean[var] == 'yes').astype(int)

    print(f"✓ Converted {len(binary_vars)} binary variables to dummy variables (1=yes, 0=no)")

    # 3. Create last digit dummy variables (1 point)
    area_last_digit = df_clean['area'].astype(int) % 10

    for digit in range(10):
        df_clean[f'end_{digit}'] = (area_last_digit == digit).astype(int)

    print("✓ Created last digit dummy variables (end_0 through end_9)")

    # Display summary of cleaning
    print(f"\nCleaning Summary:")
    print(f"- Original variables: {len(df.columns)}")
    print(f"- Variables after cleaning: {len(df_clean.columns)}")
    print(f"- New variables created: area2, {', '.join([f'end_{i}' for i in range(10)])}")

    # Show distribution of area last digits
    print("\nArea last digit distribution:")
    for digit in range(10):
        count = sum(area_last_digit == digit)
        pct = count / len(df_clean) * 100
        print(f"  end_{digit}: {count:4d} ({pct:5.1f}%)")

    return df_clean

# Perform data cleaning
df_clean = clean_data(df)


# ## Visualize Area Distribution
# 
# Let's visualize the distribution of areas and their last digits to understand the data better.

# In[30]:


# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Area distribution
axes[0,0].hist(df_clean['area'], bins=50, alpha=0.7, color='skyblue')
axes[0,0].set_title('Distribution of Apartment Areas')
axes[0,0].set_xlabel('Area (m²)')
axes[0,0].set_ylabel('Frequency')

# Last digit distribution
last_digits = df_clean['area'].astype(int) % 10
digit_counts = [sum(last_digits == i) for i in range(10)]
axes[0,1].bar(range(10), digit_counts, alpha=0.7, color='lightgreen')
axes[0,1].set_title('Distribution of Area Last Digits')
axes[0,1].set_xlabel('Last Digit')
axes[0,1].set_ylabel('Count')
axes[0,1].set_xticks(range(10))

# Price distribution
axes[1,0].hist(df_clean['price'], bins=50, alpha=0.7, color='orange')
axes[1,0].set_title('Distribution of Apartment Prices')
axes[1,0].set_xlabel('Price (PLN)')
axes[1,0].set_ylabel('Frequency')

# Price vs Area scatter
axes[1,1].scatter(df_clean['area'], df_clean['price'], alpha=0.5, color='red')
axes[1,1].set_title('Price vs Area')
axes[1,1].set_xlabel('Area (m²)')
axes[1,1].set_ylabel('Price (PLN)')

plt.tight_layout()
plt.show()

# Price statistics by last digit
print("\nPrice statistics by area last digit:")
for digit in range(10):
    mask = df_clean[f'end_{digit}'] == 1
    if sum(mask) > 0:
        avg_price = df_clean.loc[mask, 'price'].mean()
        count = sum(mask)
        print(f"  Digit {digit}: {count:4d} apartments, avg price: {avg_price:8,.0f} PLN")


# ## Part 3b: Linear Model Estimation (4 points)
# 
# We'll estimate a hedonic pricing model using two methods:
# 1. Standard linear regression
# 2. Partialling-out method (Frisch-Waugh-Lovell theorem)
# 
# Both methods should produce identical coefficients.

# In[31]:


def linear_model_estimation(df):
    """
    Perform linear model estimation as specified in Part 3b.

    Tasks:
    1. Regress price against specified covariates
    2. Perform the same regression using partialling-out method
    3. Verify coefficients match
    """
    print("\n=== LINEAR MODEL ESTIMATION (Part 3b) ===\n")

    # Prepare the feature matrix
    features = []

    # Area's last digit dummies (omit 9 to have a base category)
    digit_features = [f'end_{i}' for i in range(9)]  # end_0 through end_8
    features.extend(digit_features)

    # Area and area squared
    features.extend(['area', 'area2'])

    # Distance variables (adjust column names to match the actual dataset)
    distance_features = []
    for col in df.columns:
        if 'distance' in col.lower():
            distance_features.append(col)
    features.extend(distance_features)

    # Binary features (those we converted)
    binary_features = []
    for col in df.columns:
        if col.startswith('has') and df[col].dtype in ['int64', 'float64']:
            binary_features.append(col)
    features.extend(binary_features)

    # Categorical variables (need to be encoded)
    categorical_vars = []
    for col in ['month', 'type', 'rooms', 'ownership', 'buildingmaterial']:
        if col in df.columns:
            categorical_vars.append(col)
        elif col.replace('building', 'building') in df.columns:
            categorical_vars.append(col.replace('building', 'building'))

    # Check actual column names
    print(f"Available columns: {list(df.columns)}")
    print(f"Distance features found: {distance_features}")
    print(f"Binary features found: {binary_features}")
    print(f"Categorical variables to encode: {categorical_vars}")

    # Create dummy variables for categorical variables
    df_encoded = df.copy()

    for var in categorical_vars:
        if var in df.columns:
            dummies = pd.get_dummies(df[var], prefix=var, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            features.extend(dummies.columns.tolist())

    # Remove any features that don't exist in the dataset
    existing_features = [f for f in features if f in df_encoded.columns]
    missing_features = [f for f in features if f not in df_encoded.columns]

    if missing_features:
        print(f"\nWarning: Missing features: {missing_features}")

    features = existing_features

    # Prepare data
    X = df_encoded[features]
    y = df_encoded['price']

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Total features: {len(features)}")

    return X, y, features, df_encoded

# Prepare the data for modeling
X, y, features, df_encoded = linear_model_estimation(df_clean)


# In[32]:


# Check for missing values in X and y before regression
print("Missing values check:")
print(f"Missing values in X: {X.isnull().sum().sum()}")
print(f"Missing values in y: {y.isnull().sum()}")

if X.isnull().sum().sum() > 0:
    print("\nColumns with missing values in X:")
    missing_cols = X.columns[X.isnull().any()]
    for col in missing_cols:
        print(f"  {col}: {X[col].isnull().sum()} missing values")

    print("\nDropping rows with missing values...")
    # Create a mask for rows without any missing values
    complete_rows = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[complete_rows]
    y_clean = y[complete_rows]

    print(f"Original data: {len(X)} rows")
    print(f"After removing missing: {len(X_clean)} rows")
    print(f"Rows removed: {len(X) - len(X_clean)}")

    # Update X and y
    X = X_clean
    y = y_clean
else:
    print("No missing values found!")


# ### Method 1: Standard Linear Regression

# In[33]:


# Method 1: Standard linear regression
print("\n1. Standard Linear Regression:")
reg = LinearRegression(fit_intercept=True)
reg.fit(X, y)

r2 = r2_score(y, reg.predict(X))

print(f"R-squared: {r2:.4f}")
print(f"Intercept: {reg.intercept_:.2f}")

# Focus on end_0 coefficient
if 'end_0' in features:
    end_0_idx = features.index('end_0')
    end_0_coef = reg.coef_[end_0_idx]
    print(f"Coefficient for end_0: {end_0_coef:.2f}")
else:
    print("Warning: end_0 feature not found in features list")
    end_0_coef = None

# Create results DataFrame
results_df = pd.DataFrame({
    'feature': ['intercept'] + features,
    'coefficient': [reg.intercept_] + reg.coef_.tolist()
})

print("\nTop 10 coefficients by magnitude:")
top_coeffs = results_df.iloc[1:].copy()  # Exclude intercept
top_coeffs['abs_coeff'] = np.abs(top_coeffs['coefficient'])
top_coeffs = top_coeffs.sort_values('abs_coeff', ascending=False).head(10)

for _, row in top_coeffs.iterrows():
    print(f"  {row['feature']:20s}: {row['coefficient']:10.2f}")


# ### Method 2: Partialling-out (FWL) Method
# 
# Now let's implement the Frisch-Waugh-Lovell theorem to estimate the coefficient for `end_0` using the partialling-out method.

# In[34]:


# Method 2: Partialling-out (FWL) method for end_0
if 'end_0' in features and end_0_coef is not None:
    print("\n2. Partialling-out Method (focusing on end_0):")

    # Separate X into X1 (end_0) and X2 (all other variables)
    X1 = X[['end_0']].values  # Variable of interest
    X2_features = [f for f in features if f != 'end_0']
    X2 = X[X2_features].values  # Control variables

    # Add intercept to X2
    X2_with_intercept = np.column_stack([np.ones(len(X2)), X2])

    # Step 1: Regress y on X2 and get residuals
    reg_y_on_x2 = LinearRegression(fit_intercept=False)
    reg_y_on_x2.fit(X2_with_intercept, y)
    y_residuals = y - reg_y_on_x2.predict(X2_with_intercept)

    # Step 2: Regress X1 on X2 and get residuals
    reg_x1_on_x2 = LinearRegression(fit_intercept=False)
    reg_x1_on_x2.fit(X2_with_intercept, X1.ravel())
    x1_residuals = X1.ravel() - reg_x1_on_x2.predict(X2_with_intercept)

    # Step 3: Regress residuals
    reg_fwl = LinearRegression(fit_intercept=False)
    reg_fwl.fit(x1_residuals.reshape(-1, 1), y_residuals)
    end_0_coef_fwl = reg_fwl.coef_[0]

    print(f"Coefficient for end_0 (FWL method): {end_0_coef_fwl:.2f}")
    print(f"Coefficient for end_0 (standard method): {end_0_coef:.2f}")
    print(f"Difference: {abs(end_0_coef - end_0_coef_fwl):.6f}")
    print(f"Methods match (within 1e-6): {abs(end_0_coef - end_0_coef_fwl) < 1e-6}")

    # Store results for later use
    model_results = {
        'model': reg,
        'features': features,
        'results_df': results_df,
        'end_0_coef_standard': end_0_coef,
        'end_0_coef_fwl': end_0_coef_fwl,
        'X': X,
        'y': y,
        'df_encoded': df_encoded
    }
else:
    print("\nSkipping FWL method as end_0 feature is not available")
    model_results = {
        'model': reg,
        'features': features,
        'results_df': results_df,
        'X': X,
        'y': y,
        'df_encoded': df_encoded
    }


# ## Part 3c: Price Premium Analysis (3 points)
# 
# Now we'll analyze whether apartments with areas ending in "0" command a price premium. We'll:
# 1. Train a model excluding apartments with area ending in 0
# 2. Use this model to predict prices for all apartments
# 3. Compare actual vs predicted prices for apartments ending in 0

# In[36]:


def price_premium_analysis(df, model_results):
    """
    Analyze price premium for apartments with area ending in 0.
    Part 3c: Price premium for area that ends in 0-digit (3 points)
    """
    print("\n=== PRICE PREMIUM ANALYSIS (Part 3c) ===\n")

    # Use the cleaned data from model_results instead of the original df
    X_clean = model_results['X']
    y_clean = model_results['y']
    features = model_results['features']

    # Create a clean DataFrame from the cleaned X and y
    df_encoded = X_clean.copy()
    df_encoded['price'] = y_clean

    # Check if we have end_0 variable
    if 'end_0' not in df_encoded.columns:
        print("Warning: end_0 variable not found. Cannot perform premium analysis.")
        return None

    # Step 1: Train model excluding apartments with area ending in 0 (1.25 points)
    print("1. Training model excluding apartments with area ending in 0:")

    # Filter out apartments with area ending in 0
    mask_not_end_0 = df_encoded['end_0'] == 0
    X_train = df_encoded.loc[mask_not_end_0, features]
    y_train = df_encoded.loc[mask_not_end_0, 'price']

    print(f"   Training sample size: {len(X_train)} (excluded {sum(~mask_not_end_0)} apartments ending in 0)")

    # Train the model
    reg_no_end_0 = LinearRegression(fit_intercept=True)
    reg_no_end_0.fit(X_train, y_train)

    r2_train = r2_score(y_train, reg_no_end_0.predict(X_train))
    print(f"   R-squared on training data: {r2_train:.4f}")

    # Step 2: Predict prices for entire sample (1.25 points)
    print("\n2. Predicting prices for entire sample:")

    X_full = df_encoded[features]
    y_full = df_encoded['price']

    # Predict using the model trained without end_0 apartments
    y_pred_full = reg_no_end_0.predict(X_full)

    print(f"   Predictions generated for {len(y_pred_full)} apartments")

    # Step 3: Compare averages for apartments ending in 0 (0.5 points)
    print("\n3. Comparing actual vs predicted prices for apartments with area ending in 0:")

    # Get apartments with area ending in 0
    mask_end_0 = df_encoded['end_0'] == 1

    actual_prices_end_0 = y_full[mask_end_0]
    predicted_prices_end_0 = y_pred_full[mask_end_0]

    # Calculate averages
    avg_actual = actual_prices_end_0.mean()
    avg_predicted = predicted_prices_end_0.mean()
    premium = avg_actual - avg_predicted
    premium_pct = (premium / avg_predicted) * 100

    print(f"   Number of apartments with area ending in 0: {sum(mask_end_0)}")
    print(f"   Average actual price: {avg_actual:,.2f} PLN")
    print(f"   Average predicted price: {avg_predicted:,.2f} PLN")
    print(f"   Price premium: {premium:,.2f} PLN ({premium_pct:+.2f}%)")

    # Additional analysis
    print(f"\n   Additional Statistics:")
    print(f"   Median actual price: {actual_prices_end_0.median():,.2f} PLN")
    print(f"   Median predicted price: {np.median(predicted_prices_end_0):,.2f} PLN")
    print(f"   Standard deviation of premium: {(actual_prices_end_0 - predicted_prices_end_0).std():,.2f} PLN")

    return {
        'model_no_end_0': reg_no_end_0,
        'avg_actual': avg_actual,
        'avg_predicted': avg_predicted,
        'premium': premium,
        'premium_pct': premium_pct,
        'n_end_0': sum(mask_end_0),
        'actual_prices_end_0': actual_prices_end_0,
        'predicted_prices_end_0': predicted_prices_end_0
    }

# Perform premium analysis
premium_results = price_premium_analysis(df_clean, model_results)


# ### Statistical Significance Test

# In[37]:


if premium_results is not None:
    # Determine if apartments ending in 0 are overpriced
    premium = premium_results['premium']
    premium_pct = premium_results['premium_pct']

    print(f"\n   Conclusion:")
    if premium > 0:
        print(f"   ✓ Apartments with area ending in 0 appear to be sold at a PREMIUM")
        print(f"     of {premium:,.2f} PLN ({premium_pct:+.2f}%) above what their features suggest.")
        print(f"     This could indicate that buyers perceive 'round' areas as more desirable")
        print(f"     or that sellers use psychological pricing strategies.")
    else:
        print(f"   ✗ Apartments with area ending in 0 appear to be sold at a DISCOUNT")
        print(f"     of {abs(premium):,.2f} PLN ({abs(premium_pct):.2f}%) below what their features suggest.")

    # Statistical significance (informal test)
    actual_prices_end_0 = premium_results['actual_prices_end_0']
    predicted_prices_end_0 = premium_results['predicted_prices_end_0']

    t_stat, p_value = stats.ttest_1samp(actual_prices_end_0 - predicted_prices_end_0, 0)

    print(f"\n   Statistical Test (t-test):")
    print(f"   Null hypothesis: Mean price difference = 0")
    print(f"   t-statistic: {t_stat:.3f}")
    print(f"   p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"   ✓ The price difference is statistically significant at 5% level.")
    else:
        print(f"   ✗ The price difference is not statistically significant at 5% level.")

    # Add to results
    premium_results['t_stat'] = t_stat
    premium_results['p_value'] = p_value


# ## Visualization of Results
# 
# Let's create some visualizations to better understand the price premium effect.

# In[38]:


if premium_results is not None:
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Actual vs Predicted Prices for end_0 apartments
    actual = premium_results['actual_prices_end_0']
    predicted = premium_results['predicted_prices_end_0']

    axes[0,0].scatter(predicted, actual, alpha=0.6, color='red')
    axes[0,0].plot([predicted.min(), predicted.max()], [predicted.min(), predicted.max()], 'k--', alpha=0.75)
    axes[0,0].set_xlabel('Predicted Price (PLN)')
    axes[0,0].set_ylabel('Actual Price (PLN)')
    axes[0,0].set_title('Actual vs Predicted Prices (Area ending in 0)')
    axes[0,0].grid(True, alpha=0.3)

    # 2. Price differences (premium) distribution
    price_diff = actual - predicted
    axes[0,1].hist(price_diff, bins=20, alpha=0.7, color='green')
    axes[0,1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0,1].axvline(x=price_diff.mean(), color='blue', linestyle='-', alpha=0.7, 
                     label=f'Mean: {price_diff.mean():.0f} PLN')
    axes[0,1].set_xlabel('Price Difference (Actual - Predicted) PLN')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Distribution of Price Premiums')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 3. Average prices by last digit
    avg_prices_by_digit = []
    digits = []
    for digit in range(10):
        mask = df_clean[f'end_{digit}'] == 1
        if sum(mask) > 0:
            avg_price = df_clean.loc[mask, 'price'].mean()
            avg_prices_by_digit.append(avg_price)
            digits.append(digit)

    bars = axes[1,0].bar(digits, avg_prices_by_digit, alpha=0.7)
    bars[0].set_color('red')  # Highlight digit 0
    axes[1,0].set_xlabel('Area Last Digit')
    axes[1,0].set_ylabel('Average Price (PLN)')
    axes[1,0].set_title('Average Price by Area Last Digit')
    axes[1,0].set_xticks(digits)
    axes[1,0].grid(True, alpha=0.3)

    # 4. Count of apartments by last digit
    counts_by_digit = []
    for digit in range(10):
        count = sum(df_clean[f'end_{digit}'] == 1)
        counts_by_digit.append(count)

    bars2 = axes[1,1].bar(range(10), counts_by_digit, alpha=0.7)
    bars2[0].set_color('red')  # Highlight digit 0
    axes[1,1].set_xlabel('Area Last Digit')
    axes[1,1].set_ylabel('Count of Apartments')
    axes[1,1].set_title('Distribution of Apartments by Area Last Digit')
    axes[1,1].set_xticks(range(10))
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ## Save Results
# 
# Let's save all our results to CSV files for future reference.

# In[40]:


def save_results(df_clean, model_results, premium_results):
    """
    Save all results to files.
    """
    print("\n=== SAVING RESULTS ===\n")

    # Create output directory
    import os
    output_dir = '../output'  # Relative path to Python/output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save cleaned data
    df_clean.to_csv(f'{output_dir}/apartments_cleaned.csv', index=False)
    print("✓ Cleaned data saved to apartments_cleaned.csv")

    # Save regression results
    model_results['results_df'].to_csv(f'{output_dir}/regression_results.csv', index=False)
    print("✓ Regression results saved to regression_results.csv")

    # Save premium analysis results
    if premium_results is not None:
        premium_summary = pd.DataFrame({
            'metric': ['n_apartments_end_0', 'avg_actual_price', 'avg_predicted_price', 
                       'premium_amount', 'premium_percentage', 't_statistic', 'p_value'],
            'value': [premium_results['n_end_0'], premium_results['avg_actual'], 
                      premium_results['avg_predicted'], premium_results['premium'],
                      premium_results['premium_pct'], 
                      premium_results.get('t_stat', np.nan), 
                      premium_results.get('p_value', np.nan)]
        })

        premium_summary.to_csv(f'{output_dir}/premium_analysis.csv', index=False)
        print("✓ Premium analysis results saved to premium_analysis.csv")

    print(f"\nAll results saved to: {output_dir}")

# Save all results
save_results(df_clean, model_results, premium_results)


# ## Summary and Conclusions
# 
# Let's create a comprehensive summary of our findings.

# In[ ]:


print("\n" + "=" * 60)
print("ASSIGNMENT 1 - PART 3: HEDONIC PRICING MODEL SUMMARY")
print("=" * 60)

print(f"\n📊 DATASET OVERVIEW:")
print(f"   • Total apartments analyzed: {len(df_clean)}")
print(f"   • Variables after cleaning: {len(df_clean.columns)}")
print(f"   • Features used in model: {len(model_results['features'])}")

print(f"\n🧹 DATA CLEANING (Part 3a - 2 points):")
print(f"   ✓ Created area² variable")
print(f"   ✓ Converted binary variables (yes/no → 1/0)")
print(f"   ✓ Created area last digit dummies (end_0 through end_9)")

print(f"\n📈 MODEL ESTIMATION (Part 3b - 4 points):")
print(f"   ✓ Standard linear regression performed")
print(f"   ✓ R-squared: {r2:.4f}")
if 'end_0_coef_standard' in model_results and 'end_0_coef_fwl' in model_results:
    print(f"   ✓ FWL method implemented and verified")
    print(f"   ✓ Coefficient matching: {abs(model_results['end_0_coef_standard'] - model_results['end_0_coef_fwl']) < 1e-6}")

if premium_results is not None:
    print(f"\n💰 PRICE PREMIUM ANALYSIS (Part 3c - 3 points):")
    print(f"   • Apartments with area ending in 0: {premium_results['n_end_0']}")
    print(f"   • Average actual price: {premium_results['avg_actual']:,.0f} PLN")
    print(f"   • Average predicted price: {premium_results['avg_predicted']:,.0f} PLN")
    print(f"   • Price premium: {premium_results['premium']:,.0f} PLN ({premium_results['premium_pct']:+.2f}%)")

    if 't_stat' in premium_results and 'p_value' in premium_results:
        print(f"   • Statistical significance: p = {premium_results['p_value']:.6f}")
        significance = "Significant" if premium_results['p_value'] < 0.05 else "Not significant"
        print(f"   • Result: {significance} at 5% level")

print(f"\n🎯 KEY FINDINGS:")
if premium_results is not None and premium_results['premium'] > 0:
    print(f"   • Evidence of PSYCHOLOGICAL PRICING in real estate market")
    print(f"   • Apartments with 'round' areas (ending in 0) command a premium")
    print(f"   • Premium suggests buyers value round numbers or sellers use strategic pricing")
elif premium_results is not None:
    print(f"   • No evidence of psychological pricing premium")
    print(f"   • Apartments with areas ending in 0 do not command a premium")
else:
    print(f"   • Premium analysis could not be completed")

print(f"\n📁 OUTPUT FILES:")
print(f"   • apartments_cleaned.csv - Cleaned dataset")
print(f"   • regression_results.csv - Model coefficients")
print(f"   • premium_analysis.csv - Premium analysis results")

print(f"\n" + "=" * 60)
print("✅ PART 3 ANALYSIS COMPLETE!")
print("=" * 60)


# ## Conclusion
# 
# This analysis has successfully implemented a comprehensive hedonic pricing model using real apartment data from Poland. We have:
# 
# ### **Part 3a (2 points)**: ✅ Data Cleaning Complete
# - Created the `area²` variable for non-linear area effects
# - Converted all binary variables from text ('yes'/'no') to numeric (1/0) format
# - Generated area last digit dummy variables (`end_0` through `end_9`) to test for psychological pricing
# 
# ### **Part 3b (4 points)**: ✅ Model Estimation Complete
# - Implemented standard linear regression with comprehensive feature set
# - Applied the Frisch-Waugh-Lovell theorem using partialling-out method
# - Verified that both methods produce identical coefficients (within machine precision)
# - Achieved strong model fit with meaningful coefficient estimates
# 
# ### **Part 3c (3 points)**: ✅ Premium Analysis Complete
# - Trained a model excluding apartments with areas ending in "0"
# - Generated price predictions for all apartments using this restricted model
# - Calculated and tested the price premium for "round" area apartments
# - Performed statistical significance testing
# 
# ### **Key Economic Insights:**
# The analysis provides evidence about psychological pricing in real estate markets. If a significant premium exists for apartments with areas ending in "0", this suggests:
# 
# 1. **Buyer Psychology**: Consumers may perceive round numbers as more desirable or trustworthy
# 2. **Seller Strategy**: Real estate agents may use psychological pricing to maximize sale prices
# 3. **Market Efficiency**: The existence of such premiums indicates potential market inefficiencies
# 
# This type of analysis is valuable for:
# - **Real estate professionals** understanding pricing strategies
# - **Policymakers** assessing market functioning
# - **Researchers** studying behavioral economics in housing markets
# 
# The methodology demonstrated here (hedonic pricing with careful feature engineering and statistical testing) is a standard approach in empirical economics and can be applied to various markets where product characteristics drive pricing.
# 
# **This completes Part 3 of Assignment 1.**
