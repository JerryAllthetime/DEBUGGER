
# Cars Resale Price Prediction

>NUS 2024 Fall Semester CS5228 Kaggle Competition

>Group 24: DEBUGGER



**Note that another branch, `Yibin`, contains an extended version of data preprocessing.**

## Requirements

Ensure you have the following libraries installed:

```bash
pip install pandas numpy scikit-learn xgboost joblib category_encoders
```

## Project Structure

- `EDA.py`: Functions for Exploratory Data Analysis (EDA).
- `main.py`: Major file.
- `preprocessing.py`: Functions for preprocessing and handling missing values.
- `predictor.py`: Functions for basic model.
- `model.py`: Functions for models' hyperparameter tuning.

## Usage

The script allows for three main tasks: `EDA`, `preprocessing`, and `predict`. These tasks can be specified with the `--task` argument.

### Commands

#### Run EDA
Perform exploratory data analysis on the dataset to understand its structure and key characteristics.

```bash
python main.py --task EDA
```

#### Run Preprocessing
Process raw data by handling missing values, generating semantic features, and saving the processed data, etc.

```bash
python main.py --task preprocessing
```

#### Train and Predict
Perform hyperparameter tuning, train selected models, and make predictions using the best-performing models.

```bash
python main.py --task predict
```

## Core Functions


### `evaluate_and_compare_models_sweep(X_train, y_train)`
Compares `RandomForestRegressor`, `HistGradientBoostingRegressor`, and `XGBRegressor` models using GridSearchCV. Selects the best parameters and combines predictions using a weighted average to minimize RMSE.

### `evaluate_and_compare_models_fixed_hyper(X_train, y_train)`
Uses fixed, predefined hyperparameters to evaluate and compare model performance. Calculates RMSE and optimizes weights for combined predictions.

### `make_predictions(rf_model, hist_model, xgb_model, best_weights, X_test)`
Uses trained models and optimal weights to make predictions on test data. Outputs results to e.g. `submission_v4_80_20.csv`.

## Customizable Parameters

Hyperparameters can be customized within each function:
- `random_forest_sweep`: Customize parameters like `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`.
- `evaluate_and_compare_models_sweep`: Customize parameters for each model in `rf_param_grid`, `hist_param_grid`, and `xgb_param_grid`.
- Weight range for combined predictions is defined in `weight_range`.

## Files and Outputs

- `processed_data/`: Directory for storing processed training and testing data.
- `submission.csv`: Final predictions in CSV format.

---
