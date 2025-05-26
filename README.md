# Bayesian GP CVLoss: Gaussian Process Regression with Cross-Validated Hyperparameter Optimization

[![PyPI version](https://badge.fury.io/py/bayesian-gp-cvloss.svg)](https://badge.fury.io/py/bayesian-gp-cvloss) <!-- Placeholder for PyPI badge -->

`bayesian_gp_cvloss` is a Python package designed to simplify the process of training Gaussian Process (GP) models by finding optimal hyperparameters through Bayesian optimization (using Hyperopt) with k-fold cross-validation. The key feature of this package is its direct optimization of the cross-validated Root Mean Squared Error (RMSE), aligning the hyperparameter tuning process closely with the model's predictive performance.

This package is particularly useful for researchers and practitioners who want to apply GP models without manually tuning hyperparameters or relying solely on maximizing marginal likelihood, offering a more direct approach to achieving good generalization on unseen data.

## Core Idea

The traditional approach to training GP models often involves maximizing the log marginal likelihood of the model parameters. While effective, this doesn't always directly translate to the best predictive performance on unseen data, especially when the model assumptions are not perfectly met or when working with smaller datasets.

This library implements an alternative strategy:

1.  **Define a search space** for the GP kernel parameters (e.g., length scales, kernel variance) and likelihood parameters (e.g., noise variance).
2.  Use **Bayesian optimization (Hyperopt)** to intelligently search this space.
3.  For each set of hyperparameters evaluated by Hyperopt, perform **k-fold cross-validation** on the training data.
4.  The **objective function** for Hyperopt is the mean RMSE across these k folds.
5.  The set of hyperparameters yielding the **minimum average cross-validated RMSE** is selected as optimal.
6.  A final GP model is then **refitted on the entire training dataset** using these best-found hyperparameters.

This method directly targets the minimization of prediction error, which can be a more robust approach for many real-world regression tasks.

## Features

*   Automated hyperparameter optimization for GP models using Hyperopt.
*   Cross-validation (k-fold) integrated into the optimization loop to find parameters that generalize well.
*   Directly optimizes for mean cross-validated RMSE.
*   Supports various GPflow kernels (e.g., RBF, Matern32, Matern52, RationalQuadratic by default, easily extensible).
*   Data-dependent default hyperparameter search space generation based on the target variable's statistics.
*   Handles mean centering of the target variable internally for potentially improved stability.
*   Simple API: provide your preprocessed numerical `X_train` and `y_train` data.

## Installation

```bash
pip install bayesian-gp-cvloss
```

Alternatively, to install the latest version directly from the source (e.g., for development):

```bash
# Ensure you have git installed
# git clone https://github.com/Shifa-Zhong/bayesian-gp-cvloss.git
# cd bayesian-gp-cvloss
pip install .
```

## Dependencies

*   gpflow >= 2.0.0
*   hyperopt >= 0.2.0
*   scikit-learn >= 0.23.0
*   pandas >= 1.0.0
*   numpy >= 1.18.0

Users are responsible for their own data preprocessing (e.g., encoding categorical features, feature scaling) before using this library. The optimizer expects purely numerical `X_train` and `y_train` inputs.

## Quick Start

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from bayesian_gp_cvloss.optimizer import GPCrossValidatedOptimizer

# 0. (User Responsibility) Load and Preprocess Data
# Example: Assume you have X (features) and y (target) as pandas DataFrames/Series
# Ensure X is purely numerical. All encoding and scaling is up to the user.

# Let's create some synthetic data for demonstration
np.random.seed(42)
N_train = 100
N_features = 3
X_synth = np.random.rand(N_train, N_features)
# A simple function for y with some noise
y_synth = np.sin(X_synth[:, 0] * 2 * np.pi) + X_synth[:, 1]**2 + np.random.randn(N_train) * 0.1

# Convert to pandas DataFrame/Series if your data isn't already
X_df = pd.DataFrame(X_synth, columns=[f'feature_{i}' for i in range(N_features)])
y_series = pd.Series(y_synth, name='target')

# Split data (optional, but good practice to have a final test set)
# The optimizer does its own CV on the X_train_opt, y_train_opt
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(
    X_df, y_series, test_size=0.2, random_state=42
)

# Scale features (example - user should choose appropriate scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_data)
X_test_scaled = scaler.transform(X_test_data)

# Optimizer expects numpy arrays
X_train_np = X_train_scaled
y_train_np = y_train_data.values

X_test_np = X_test_scaled
y_test_np = y_test_data.values

# 1. Initialize the Optimizer
# You can specify kernels, number of folds, max_evals for Hyperopt, etc.
optimizer = GPCrossValidatedOptimizer(
    n_folds=5, 
    max_evals=50, # Number of Hyperopt trials (increase for better results)
    random_state_hyperopt=42
)

# 2. Run Optimization
# This will find the best hyperparameters based on cross-validated RMSE
# using X_train_np and y_train_np
optimizer.optimize(X_train_np, y_train_np)

print(f"Best hyperparameters found: {optimizer.best_params}")
print(f"Best CV validation RMSE: {optimizer.best_cv_val_rmse_}")
print(f"Best CV train RMSE: {optimizer.best_cv_train_rmse_}")

# 3. Get the Refitted Model (Optional)
# The optimizer automatically refits a model on the full X_train_np, y_train_np 
# using the best hyperparameters. You can access it if needed.
# best_gpr_model = optimizer.get_refitted_model()
# print_summary(best_gpr_model) # Requires gpflow

# 4. Make Predictions
# The predict method uses the refitted model.
# Input to predict should be preprocessed in the same way as X_train_np
y_pred_test, y_pred_var_test = optimizer.predict(X_test_np)

# Evaluate (example)
from sklearn.metrics import mean_squared_error
rmse_test = np.sqrt(mean_squared_error(y_test_np, y_pred_test))
print(f"Test RMSE: {rmse_test}")

# Plot results (example)
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(y_test_np, y_pred_test, alpha=0.7, label='Test Predictions')
plt.plot([min(y_test_np), max(y_test_np)], [min(y_test_np), max(y_test_np)], 'r--', label='Ideal')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("GPR Predictions vs True Values on Test Set")
plt.legend()
plt.grid(True)
plt.show()

```

## How it Works Internally

1.  **`__init__(...)`**: Initializes settings like number of folds (`n_folds`), Hyperopt maximum evaluations (`max_evals`), desired kernels (`kernels_to_try`), and random states.
2.  **`optimize(X_train, y_train)`**:
    *   Stores `X_train` and `y_train`.
    *   Calculates `self.y_train_mean_` for internal centering.
    *   Calls `_get_default_data_dependent_space(y_train)` to define the hyperparameter search space for Hyperopt. This space is dynamically adjusted based on the variance and standard deviation of `y_train` to provide sensible default ranges for kernel variance and likelihood noise.
    *   Initializes `hyperopt.Trials()`.
    *   Runs `hyperopt.fmin()` with the `_objective` function, the defined `space`, `tpe.suggest` algorithm, and `max_evals`.
    *   Stores the best parameters (`self.best_params`), best cross-validated validation RMSE (`self.best_cv_val_rmse_`), and corresponding training RMSE (`self.best_cv_train_rmse_`).
    *   Calls `refit_best_model(X_train, y_train)` to train a final GPR model on the full training data using `self.best_params`.
3.  **`_objective(params)`**:
    *   This is the function minimized by Hyperopt.
    *   It takes a dictionary of `params` (hyperparameters for a single trial).
    *   Performs k-fold cross-validation:
        *   For each fold, splits `X_train`, `y_train` into `X_train_fold`, `y_train_fold` and `X_val_fold`, `y_val_fold`.
        *   **Important**: `y_train_fold` and `y_val_fold` are centered by subtracting the mean of the *current* `y_train_fold`.
        *   Constructs a GPflow GPR model using the hyperparameters from `params` and the current fold's training data.
        *   Predicts on `X_val_fold` and calculates RMSE.
    *   Averages the RMSEs from all validation folds.
    *   Returns a dictionary including `{'loss': avg_val_rmse, 'status': STATUS_OK, ...}`.
4.  **`_get_default_data_dependent_space(y_train)`**: 
    *   Defines the search space for Hyperopt for each hyperparameter:
        *   `lengthscales`: `hp.quniform` between 1 and 100 (step 0.01) for each input dimension.
        *   `kernel_variance`: `hp.uniform` between 0 and `y_train.var()`.
        *   `likelihood_noise_variance`: `hp.loguniform` between `(y_train.std()/100)**2` and `(y_train.std()/2)**2` (with safety checks for small/zero std dev).
        *   `kernel_class`: `hp.choice` among the kernels specified in `self.kernels_to_try`.
5.  **`refit_best_model(X_data_refit, y_data_refit)`**:
    *   Trains a new GPflow GPR model using `self.best_params` on the *entire* `X_data_refit` and `y_data_refit` (which are centered using `self.y_train_mean_`).
    *   Stores this model as `self.final_model_`.
6.  **`predict(X_new)`**:
    *   Takes new, preprocessed data `X_new`.
    *   Uses `self.final_model_` to predict mean and variance.
    *   Adds back `self.y_train_mean_` to the predicted mean to return predictions on the original scale.

## Customization

*   **Kernels**: Pass a list of GPflow kernel classes to the `kernels_to_try` argument in the `GPCrossValidatedOptimizer` constructor (e.g., `[gpflow.kernels.Matern52, gpflow.kernels.RBF]`).
*   **Hyperparameter Space**: While a data-dependent default space is provided, you can supply your own `hyperopt_space` dictionary to the `optimize` method if you need finer control or different distributions for hyperparameters.
*   **Cross-Validation**: Change `n_folds` and `random_state_kfold`.
*   **Hyperopt**: Adjust `max_evals` and `random_state_hyperopt`.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please open an issue or submit a pull request to the GitHub repository: https://github.com/Shifa-Zhong/bayesian-gp-cvloss

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Shifa Zhong (sfzhong@tongji.edu.cn)
GitHub: [Shifa-Zhong](https://github.com/Shifa-Zhong) 