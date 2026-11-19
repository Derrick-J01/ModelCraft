# Model 1.0 — Ames Housing Price Prediction

## Objective
Predict residential home sale prices using the Kaggle House Prices: Advanced Regression Techniques dataset.

## Workflow
1. Load and inspect the data using pandas.
2. Separate SalePrice from the input features.
3. Preprocess numerical, nominal, and ordinal features.
4. Evaluate a Random Forest using five-fold cross-validation.
5. Fit the final model on all labeled training data.
6. Generate predictions for the competition test dataset.

## Features
- Numerical: LotArea, YearBuilt, PoolArea
- Nominal: Neighborhood (one-hot encoding)
- Ordinal: ExterQual, KitchenQual, HeatingQC (quality order: Po, Fa, TA, Gd, Ex)

## Current evaluation
Five-fold cross-validation mean absolute error: $24,898.38.

See metrics.md for the fold-level results and comparison notes.
