# London Model Run Report

## Baseline Summary

- Best model: `HistGBR`
- Test RMSE: `393489.27`
- Test MAE: `216082.64`
- RMSE improvement vs naive: `57.58%`
- Test R²: `0.7197`
- Test MAPE: `0.3381`
- Share of test rows with |y−ŷ|/|y| < 10% (|y|>100 GBP): `24.27%`
- Price-bin accuracy (5 quantile bins, `HistGBR`): `49.51%`
- Price-bin macro F1: `0.4950`

## Walk-Forward Stability

### RandomForest
- Folds: `4`
- Mean RMSE: `347500.18`
- RMSE std: `84774.58`
- Coefficient of variation: `24.40%`

### MLP
- Folds: `4`
- Mean RMSE: `356658.29`
- RMSE std: `53909.12`
- Coefficient of variation: `15.12%`

## External Estimate Benchmark

- Best benchmark channel: `saleEstimate_lowerPrice`
- Benchmark RMSE: `350168.21`
- Rows used: `130196`

## Assisted Track

- Best assisted model: `AssistedHistGBR`
- Test RMSE: `304436.22`
- Test MAE: `158399.33`

## Cross-Track Comparison

- Mainline RMSE: `393489.27`
- Assisted RMSE: `304436.22`
- External benchmark RMSE: `350168.21`
- Preferred track by RMSE: `assisted`

## Decision Summary

- Best primary RMSE: `393489.27`
- Best benchmark RMSE: `350168.21`
- Delta (primary - benchmark): `43321.06`
- Recommendation: `open_assisted_track`
- Governance recommendation: `deploy_assisted`

## Release Gates

- No crash across baseline + walk-forward + benchmark: `PASS`
- Best primary RMSE <= 396614.68: `PASS`
- Walk-forward CV <= 26%: `PASS`
- High-support segment RMSE <= 1.8x overall (708280.69): `FAIL`

## Blockers

- High-support segments above threshold: propertyType:Semi-Detached Property (HistGBR, rmse=825627.50, rows=223); price_band:(1000000.0, 4400000.0] (HistGBR, rmse=768250.59, rows=26061); propertyType:Detached House (HistGBR, rmse=746635.17, rows=3733)
