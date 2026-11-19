# Model 2.0 --- Ames Housing Price Prediction

## Overview

Model 2.0 is the next iteration of the ModelCraft Ames Housing
price-prediction project. It uses **XGBoost regression** to predict
residential sale prices from the Kaggle Ames Housing dataset.

The project has progressed from an initial Random Forest model in Model
1.0 to gradient-boosted decision trees in Model 2.0. The goal is to
improve predictive performance while building a clearer, more systematic
modeling workflow.

## Progress from Model 1.0

Model 1.0 established the initial modeling workflow and explored feature
selection, preprocessing, and Random Forest regression. The
expanded-feature Random Forest run recorded:

  Metric                                             Model 1.0
  --------------------------------------------- --------------
  Mean Absolute Error (MAE)                       `$24,537.67`
  Root Mean Squared Logarithmic Error (RMSLE)        `0.17623`

Model 2.0 moved to XGBoost, a gradient-boosted decision-tree method in
which trees are added sequentially to improve the ensemble's
predictions.

The latest Model 2.0 validation run recorded:

  Metric                                             Model 2.0
  --------------------------------------------- --------------
  Mean Absolute Error (MAE)                       `$22,231.13`
  Root Mean Squared Logarithmic Error (RMSLE)        `0.16014`

Relative to the recorded Model 1.0 results, these figures represent
approximately **9.4% lower MAE** and **9.1% lower RMSLE**.

These are validation results, not Kaggle leaderboard scores. Treat the
comparison as indicative of progress; a strict model-to-model comparison
requires the same validation split and evaluation procedure for both
versions.

## Dataset

This project uses the Kaggle **House Prices: Advanced Regression
Techniques** dataset.

Place these files in the project directory before running the model:

-   `train.csv`
-   `test.csv`

The CSV files are not included in this repository. Obtain them from the
Kaggle competition page.

## Model and preprocessing

The model is implemented in `model2.py`. It uses XGBoost and
scikit-learn tools for model training, preprocessing, and evaluation.

The preprocessing workflow handles feature types such as:

-   Numerical features
-   Ordinal categorical features
-   Nominal categorical features using one-hot encoding

Preprocessing should be fitted on the training data and then applied to
validation or test data to avoid information leakage.

See `model2.py` for the exact feature list, preprocessing configuration,
model hyperparameters, and training/validation procedure.

## Evaluation

The current validation results are:

-   **MAE:** approximately `$22,231`
-   **RMSLE:** approximately `0.16014`

MAE expresses the average absolute prediction error in the target's
units (dollars). RMSLE evaluates error on a logarithmic scale,
emphasizing relative differences between actual and predicted prices.

Results can vary if the data split, features, preprocessing, or model
settings change.

## Next steps

The next phase of the project will focus on **feature engineering**.
This includes investigating how existing variables can be transformed,
combined, or represented more effectively to capture additional
information about house prices.

Potential areas to explore include:

-   Creating features that represent house age or remodeling age.
-   Combining related area measurements into useful total-area features.
-   Exploring interactions between overall quality, living area, and
    neighborhood.
-   Reviewing skewed numerical variables and considering
    transformations.
-   Testing whether engineered features improve validation performance.

New features will be evaluated using a consistent validation approach to
determine whether they provide genuine predictive value.

## Setup

Create and activate a virtual environment:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

``` bash
python -m pip install pandas scikit-learn xgboost
```

If XGBoost reports a missing OpenMP runtime on macOS, install it with
Homebrew:

``` bash
brew install libomp
```

## Run

With `train.csv` and `test.csv` in the project directory and the virtual
environment activated:

``` bash
python model2.py
```

## Project files

-   `model2.py` --- main model implementation.
-   `model2.pyx` --- accompanying source file; see the file itself for
    its specific role.
-   `README.md` --- project overview, progress, evaluation, and next
    steps.

## Notes

-   The `.venv/` directory is a local environment and should not be
    committed.
-   Dataset files should generally remain outside version control.
-   Update the evaluation figures when the model or validation procedure
    changes.
