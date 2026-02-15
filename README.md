# Water-Injection-Machine-Learning
Machine Learning project to predict Injectivity Rate in water injection wells and analyze key factors affecting well performance in the oil &amp; gas industry

## Project Overview

Water injection is a widely used secondary recovery technique in the oil and gas industry to maintain reservoir pressure and improve oil recovery. Predicting the **injectivity rate** -- the volume of water injected per day -- is critical for:

- Optimizing injection schedules and volumes
- Detecting early signs of formation damage or wellbore issues
- Reducing operational downtime and costs
- Planning maintenance and workover operations

This project applies supervised machine learning models to historical monthly well data spanning **1998--2019** to forecast injectivity rates and identify the key operational factors that drive injection performance.

## Data Description

The dataset contains **248 monthly records** with the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| `Month/year` | Date of the observation (monthly) | datetime |
| `Working hours, h` | Total operational hours in the month | hours |
| `Injection volume, m3` | Total volume of water injected | cubic meters |
| `Well downtime, h` | Total hours the well was non-operational | hours |
| `Injectivity rate (m3/day)` | **Target variable** -- daily injection rate | m3/day |

### Sample Data

| Month/year | Working hours, h | Injection volume, m3 | Well downtime, h | Injectivity rate (m3/day) |
|------------|-----------------|---------------------|------------------|--------------------------|
| 2018-12-01 | 744 | 6241 | 0 | 201.32 |
| 2018-11-01 | 720 | 6769 | 0 | 225.63 |
| 2018-10-01 | 72 | 711 | 672 | 237.00 |
| 2018-09-01 | 362 | 6562 | 358 | 435.05 |
| 2018-08-01 | 170 | 2945 | 574 | 415.76 |

## Methodology

### Step 1: Exploratory Data Analysis (EDA)
- Time-series visualization of injectivity rate
- Rolling mean and standard deviation analysis (12-month window)
- Seasonal decomposition (additive model, period=12)
- Correlation heatmap between all features

### Step 2: Feature Engineering
- **Temporal features**: month, quarter, year
- **Lag features**: 1-month, 2-month, 3-month lagged injectivity rates
- **Rolling statistics**: 3-month rolling mean squared values
- **Operational ratios**: injection per hour, downtime ratio
- **Interaction terms**: volume x working hours

### Step 3: Outlier Detection & Data Cleaning
- Z-score filtering (threshold = 3)
- IQR-based outlier removal
- Result: 243 clean records from 245 (after feature engineering)

### Step 4: Model Training & Validation
Three models were trained and evaluated using **TimeSeriesSplit** cross-validation:

| Model | RMSE | MAE | R2 |
|-------|------|-----|-----|
| Linear Regression | 37.97 | 23.43 | 0.73 |
| XGBoost | 49.07 | 20.72 | 0.44 |
| **XGBoost (Tuned)** | -- | -- | **0.63** |

### Step 5: Hyperparameter Tuning
- RandomizedSearchCV with TimeSeriesSplit
- Best parameters: `subsample=0.9, n_estimators=200, max_depth=3, learning_rate=0.1, colsample_bytree=0.8`
- Tuned R2: **0.6280**

### Step 6: Uncertainty Quantification
- Quantile regression using GradientBoostingRegressor
- 90% prediction interval (5th--95th percentile)

### Step 7: Explainable AI (SHAP)
- SHAP TreeExplainer for feature importance
- SHAP summary plot showing impact of each feature
- Partial Dependence Plots for top features

### Step 8: Scenario Simulation
- Function to simulate injectivity under different operational conditions
- Input: working hours, injection volume, downtime
- Output: predicted injectivity rate

### Step 9: Out-of-Sample Validation
- Last 12 months held out as test set
- Model retrained on all prior data
- Final evaluation on unseen data

## Key Results

- **Linear Regression** achieved the best cross-validated R2 (0.73), indicating strong linear relationships in the data
- **XGBoost** after tuning improved from R2=0.44 to R2=0.63
- **SHAP analysis** revealed that `Working hours`, `Injection volume`, and `lag` features are the most influential predictors
- The **90% prediction interval** provides reliable uncertainty bounds for operational planning


*This project was developed as part of a data science portfolio focused on applying machine learning to real-world oil & gas engineering problems.*


