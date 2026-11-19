# Model 1.0 — Performance Log

## Random Forest baseline
- Features: LotArea, YearBuilt, PoolArea
- Evaluation: 75/25 train-validation split
- Random state: 0
- MAE: $38,265.96

## Random Forest with expanded features
- Features: LotArea, YearBuilt, PoolArea, Neighborhood, ExterQual, KitchenQual, HeatingQC
- Evaluation: 5-fold cross-validation

### Fold MAE scores
- $22,994.41
- $27,089.72
- $25,258.13
- $23,222.24
- $25,927.38

- Average MAE: $24,898.38
- Minimum MAE: $22,994.41
- Maximum MAE: $27,089.72
- Standard deviation: $1,576.37

## Comparison note
The baseline and expanded-feature results use different feature sets, preprocessing, and evaluation methods. This is not a controlled comparison isolating the effect of any one change.

## Next step
Calculate RMSLE using out-of-fold predictions and record the result here.
