# Bayesian GP CVLoss: Gaussian Process Regression with Cross-Validated Hyperparameter Optimization

[![PyPI version](https://badge.fury.io/py/bayesian-gp-cvloss.svg)](https://badge.fury.io/py/bayesian-gp-cvloss)

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
git clone https://github.com/Shifa-Zhong/bayesian-gp-cvloss.git
cd bayesian-gp-cvloss
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

from bayesian_gp_cvloss import GPCrossValidatedOptimizer

# 0. (User Responsibility) Load and Preprocess Data
# Ensure X is purely numerical. All encoding and scaling is up to the user.

# Create some synthetic data for demonstration
np.random.seed(42)
N_train = 100
N_features = 3
X_synth = np.random.rand(N_train, N_features)
y_synth = np.sin(X_synth[:, 0] * 2 * np.pi) + X_synth[:, 1]**2 + np.random.randn(N_train) * 0.1

X_df = pd.DataFrame(X_synth, columns=[f'feature_{i}' for i in range(N_features)])
y_series = pd.Series(y_synth, name='target')

# Split data
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(
    X_df, y_series, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_data)
X_test_scaled = scaler.transform(X_test_data)

y_train_np = y_train_data.values

# 1. Initialize the Optimizer
# Pass preprocessed X_train and y_train directly to the constructor.
# A data-dependent default hyperparameter search space is generated automatically.
optimizer = GPCrossValidatedOptimizer(
    X_train=X_train_scaled,
    y_train=y_train_np,
    n_splits=5,          # Number of CV folds
    random_state=42       # For reproducibility
)

# 2. Run Optimization
# This finds the best hyperparameters based on cross-validated RMSE
# and automatically refits a final model on the full training data.
best_params = optimizer.optimize(max_evals=50)

print(f"Best hyperparameters found: {best_params}")

# Access the best trial's CV RMSE from the trials object
trials = optimizer.get_optimization_results()
if trials.best_trial:
    print(f"Best CV RMSE: {trials.best_trial['result']['loss']:.4f}")
    print(f"Best CV Train RMSE: {trials.best_trial['result']['train_loss']:.4f}")

# 3. Make Predictions
# The predict method uses the refitted model and returns predictions
# on the original (uncentered) scale.
y_pred_test, y_pred_var_test = optimizer.predict(X_test_scaled)

# 4. Evaluate
from sklearn.metrics import mean_squared_error
rmse_test = np.sqrt(mean_squared_error(y_test_data.values, y_pred_test))
print(f"Test RMSE: {rmse_test:.4f}")
```

## How it Works Internally

1.  **`__init__(X_train, y_train, hyperopt_space=None, n_splits=5, random_state=None)`**: Stores the preprocessed training data, computes `y_train_mean_` for internal centering, and generates a data-dependent default hyperparameter search space if `hyperopt_space` is not provided.
2.  **`optimize(max_evals=100, tpe_algo=tpe.suggest, early_stop_fn=None, rstate_seed=None)`**:
    *   Initializes `hyperopt.Trials()`.
    *   Runs `hyperopt.fmin()` with the `_objective` function, the defined search space, `tpe.suggest` algorithm, and `max_evals`.
    *   Stores the best parameters in `self.best_params`.
    *   Calls `refit_best_model()` to train a final GPR model on the full training data using `self.best_params`.
    *   Returns `self.best_params`.
3.  **`_objective(params)`**:
    *   This is the function minimized by Hyperopt.
    *   It takes a dictionary of `params` (hyperparameters for a single trial).
    *   Performs k-fold cross-validation:
        *   For each fold, splits `X_train`, `y_train` into training and validation subsets.
        *   **Important**: The target variable in each fold is centered by subtracting the mean of the *current fold's* training target.
        *   Constructs a GPflow GPR model using the hyperparameters from `params` and the current fold's training data.
        *   Predicts on the validation fold and calculates RMSE.
    *   Averages the RMSEs from all validation folds.
    *   Returns a dictionary including `{'loss': avg_val_rmse, 'status': STATUS_OK, ...}`.
4.  **`_get_default_data_dependent_space()`**:
    *   Defines the search space for Hyperopt for each hyperparameter:
        *   `lengthscales_{i}`: `hp.quniform` between 0.1 and 100 (step 0.01) for each input dimension.
        *   `kernel_variance`: `hp.uniform` between 1e-6 and `y_train.var()`.
        *   `likelihood_noise_variance`: `hp.loguniform` between `(y_train.std()/100)**2` and `(y_train.std()/2)**2` (with safety checks for small/zero std dev).
        *   `kernel_name`: `hp.choice` among the default kernels (Matern32, Matern52, RBF, RationalQuadratic).
5.  **`refit_best_model()`**:
    *   Trains a new GPflow GPR model using `self.best_params` on the *entire* training data (centered using `self.y_train_mean_`).
    *   Stores this model as `self.best_model_`.
6.  **`predict(X_new_processed)`**:
    *   Takes new, preprocessed data `X_new_processed`.
    *   Uses `self.best_model_` to predict mean and variance.
    *   Adds back `self.y_train_mean_` to the predicted mean to return predictions on the original scale.
    *   Returns `(pred_mean, pred_var)` as NumPy arrays.

## Customization

*   **Kernels**: Modify `DEFAULT_KERNELS` in `bayesian_gp_cvloss.optimizer` or provide a custom `hyperopt_space` with your desired `kernel_name` choices.
*   **Hyperparameter Space**: Pass a custom `hyperopt_space` dictionary to the `GPCrossValidatedOptimizer` constructor. The space must include keys for `lengthscales_{i}` (for each feature), `kernel_variance`, `likelihood_noise_variance`, and `kernel_name`.
*   **Cross-Validation**: Change `n_splits` and `random_state` in the constructor.
*   **Hyperopt**: Adjust `max_evals` and `rstate_seed` in the `optimize()` method.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please open an issue or submit a pull request to the GitHub repository: https://github.com/Shifa-Zhong/bayesian-gp-cvloss

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Shifa Zhong (sfzhong@tongji.edu.cn)
GitHub: [Shifa-Zhong](https://github.com/Shifa-Zhong)
