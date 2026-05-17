# Bayesian GP CVLoss: Gaussian Process Regression with Cross-Validated Hyperparameter Optimization

[![PyPI version](https://badge.fury.io/py/bayesian-gp-cvloss.svg)](https://badge.fury.io/py/bayesian-gp-cvloss)

`bayesian_gp_cvloss` is a Python package designed to simplify the process of training Gaussian Process (GP) models by finding optimal hyperparameters through Bayesian optimization (using Hyperopt) with k-fold cross-validation. The key feature of this package is its direct optimization of the cross-validated loss, aligning the hyperparameter tuning process closely with the model's predictive performance.

This package is particularly useful for researchers and practitioners who want to apply GP models without manually tuning hyperparameters or relying solely on maximizing marginal likelihood, offering a more direct approach to achieving good generalization on unseen data.

## Core Idea

The traditional approach to training GP models often involves maximizing the log marginal likelihood of the model parameters. While effective, this doesn't always directly translate to the best predictive performance on unseen data, especially when the model assumptions are not perfectly met or when working with smaller datasets.

This library implements an alternative strategy:

1.  **Define a search space** for the GP kernel parameters (e.g., length scales, kernel variance) and likelihood parameters (e.g., noise variance).
2.  Use **Bayesian optimization (Hyperopt)** to intelligently search this space.
3.  For each set of hyperparameters evaluated by Hyperopt, perform **k-fold cross-validation** on the training data.
4.  The **objective function** is configurable: cross-validated RMSE, Negative Log Predictive Density (NLPD), or a weighted combination.
5.  The set of hyperparameters yielding the **minimum loss** is selected as optimal.
6.  A final GP model is then **refitted on the entire training dataset** using these best-found hyperparameters.

## Features

*   Automated hyperparameter optimization for GP models using Hyperopt.
*   Cross-validation (k-fold) integrated into the optimization loop.
*   **Three scoring objectives**:
    - `"cv_rmse"` — Minimise cross-validated RMSE (prediction accuracy).
    - `"nlpd"` — Minimise Negative Log Predictive Density (prediction accuracy + uncertainty calibration).
    - `"combined"` — Weighted combination of both, balancing accuracy and calibration.
*   **Post-hoc variance calibration** (`ConformalCalibrator`, v0.3.0+): pair with `scoring="cv_rmse"` to get RMSE-optimal point predictions plus prediction intervals with a finite-sample marginal coverage guarantee, via locally adaptive split conformal prediction.
*   **Automatic Leave-One-Out (LOO)**: when the dataset is smaller than `n_splits`, the splitter falls back to LOO automatically.
*   Supports various GPflow kernels (RBF, Matern32, Matern52, RationalQuadratic by default).
*   **Smart data-dependent defaults**: search ranges are automatically computed from the training data.
*   **Flexible overrides**: fine-tune individual search ranges without building a full Hyperopt space.
*   Simple API: provide your preprocessed numerical `X_train` and `y_train` data.

## Installation

```bash
pip install bayesian-gp-cvloss
```

Alternatively, install from source:

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

## Quick Start

```python
import numpy as np
from bayesian_gp_cvloss import GPCrossValidatedOptimizer

# Create synthetic data
np.random.seed(42)
X = np.random.rand(100, 3)
y = np.sin(X[:, 0] * 2 * np.pi) + X[:, 1]**2 + np.random.randn(100) * 0.1

# --- Option A: Classic RMSE objective (default, backward-compatible) ---
optimizer = GPCrossValidatedOptimizer(
    X_train=X, y_train=y,
    n_splits=5, random_state=42
)
best_params = optimizer.optimize(max_evals=50)

# --- Option B: NLPD objective (accuracy + uncertainty calibration) ---
optimizer = GPCrossValidatedOptimizer(
    X_train=X, y_train=y,
    scoring="nlpd",           # <-- NEW
    n_splits=5, random_state=42
)
best_params = optimizer.optimize(max_evals=50)

# --- Option C: Combined objective ---
optimizer = GPCrossValidatedOptimizer(
    X_train=X, y_train=y,
    scoring="combined",       # <-- NEW
    nlpd_weight=0.5,          # <-- NEW: weight for NLPD term
    n_splits=5, random_state=42
)
best_params = optimizer.optimize(max_evals=50)

# Access results — both RMSE and NLPD are always recorded
trials = optimizer.get_optimization_results()
if trials.best_trial:
    result = trials.best_trial['result']
    print(f"Best CV RMSE: {result['cv_rmse']:.4f}")
    print(f"Best CV NLPD: {result['cv_nlpd']:.4f}")
    print(f"Best Train RMSE: {result['train_loss']:.4f}")

# Predict
y_pred, y_var = optimizer.predict(X_test)
```

## Scoring Objectives Explained

### `"cv_rmse"` (default)

Minimises the mean cross-validated Root Mean Squared Error. This directly targets prediction accuracy and is equivalent to the behaviour of v0.1.x.

### `"nlpd"` — Negative Log Predictive Density

Treats the GP prediction as a Gaussian distribution N(mu, sigma^2) and evaluates how likely the true observation is under that distribution:

```
NLPD = 0.5 * log(2*pi) + 0.5 * log(sigma^2) + 0.5 * (y - mu)^2 / sigma^2
```

This simultaneously penalises:
- **Inaccurate means**: large `(y - mu)^2`
- **Overconfident predictions**: small `sigma^2` when the prediction is wrong
- **Underconfident predictions**: large `sigma^2` when the prediction is right

This is particularly important for Bayesian optimisation, where acquisition functions (EI, UCB, etc.) depend on both the predicted mean and variance.

### `"combined"`

A weighted sum of normalised RMSE and NLPD:

```
loss = (1 - nlpd_weight) * norm_RMSE + nlpd_weight * norm_NLPD
```

Both metrics are min-max normalised using the optimisation history so that the weight is meaningful regardless of scale. The default `nlpd_weight=0.5` gives equal importance to accuracy and calibration.

## Post-hoc Variance Calibration with `ConformalCalibrator`

NLPD-style joint optimisation pushes the GP toward calibrated variance at the cost of some mean accuracy. If you would rather **keep the RMSE-optimal mean** and fix the variance separately, use `ConformalCalibrator` on top of a `scoring="cv_rmse"` model.

It implements **locally adaptive split conformal prediction**: it does not retrain the GP and makes no Gaussian assumption — it learns a single multiplier `q` from a held-out calibration set so that the intervals `[mu - q*sigma, mu + q*sigma]` have marginal coverage `>= 1 - alpha`.

```python
from bayesian_gp_cvloss import GPCrossValidatedOptimizer, ConformalCalibrator

# 1. Train the GP with the default cv_rmse objective for best point predictions
optimizer = GPCrossValidatedOptimizer(X_train, y_train, scoring="cv_rmse")
optimizer.optimize(max_evals=50)

# 2a. Calibrate on a HELD-OUT set (must be disjoint from training data)
calibrator = ConformalCalibrator(alpha=0.1).fit(optimizer, X_cal, y_cal)

# 3. Get prediction intervals with >= 90% marginal coverage
mean, lower, upper = calibrator.predict_interval(X_new)

# Or get the calibrated half-width q*sigma directly
half_width = calibrator.calibrated_half_width(X_new)
```

### No held-out data? Use `fit_cv` instead

When data is scarce (typical for materials BO), splitting off a calibration set is wasteful. `fit_cv` runs K-fold (or LOO) cross-validation with the **already-selected** best hyperparameters and uses those out-of-fold residuals as calibration scores. No extra data needed.

```python
optimizer = GPCrossValidatedOptimizer(X_train, y_train, scoring="cv_rmse", n_splits=5)
optimizer.optimize(max_evals=50)

# OOF-based calibration -- no separate X_cal/y_cal required
calibrator = ConformalCalibrator(alpha=0.1).fit_cv(optimizer)
mean, lower, upper = calibrator.predict_interval(X_new)
```

**Why not reuse the CV residuals from inside the hyperopt search?** Those residuals are *post-selection biased*: hyperopt picked the trial with the smallest CV-RMSE, so its residuals systematically underestimate generalisation error. `fit_cv` re-runs CV **after** hyperparameters are frozen, which removes that bias.

**Coverage trade-off**: `fit_cv` implements naive cross-conformal prediction. The formal 1-alpha split-CP proof does not strictly apply (calibration uses K fold models; test-time prediction uses the global model trained on all data), so coverage is approximately rather than exactly 1-alpha. In practice it tracks nominal coverage closely. Use `fit` with a held-out set if you need the exact guarantee.

### How it works

For each calibration point, compute the normalised residual `s_i = |y_i - mu_i| / sigma_i`. The conformal quantile is the `ceil((n+1)*(1-alpha))`-th smallest score, where the `(n+1)` term is the finite-sample correction that makes the coverage guarantee exact (not just asymptotic).

Because the score divides by `sigma_i`, the GP's per-point uncertainty *ordering* is preserved — points where the GP says it is uncertain still get wider intervals. The calibrator only fixes the overall scale.

### When to use which

| Goal | Recommended approach |
| --- | --- |
| Most accurate point predictions + valid prediction intervals | `scoring="cv_rmse"` + `ConformalCalibrator` |
| Single-pass training, no held-out data to spare | `scoring="nlpd"` or `"combined"` |
| Bayesian optimisation (acquisition needs proper posterior) | `scoring="nlpd"` or `"combined"` |

### Caveats

*   `X_cal` / `y_cal` **must be disjoint** from the training set, otherwise exchangeability breaks and intervals become over-optimistic.
*   The guarantee is **marginal**, not conditional: averaged over inputs, coverage is `>= 1 - alpha`, but it may be loose in easy regions and tight in hard ones.
*   The calibration set must be large enough for the `(n+1)` quantile to exist, i.e. `n >= ceil(1/alpha)`. For `alpha=0.1` that means `n >= 10`.
*   In active learning / BO loops, newly selected points are not exchangeable with the calibration set, so re-calibrate periodically rather than relying on a one-shot fit.

## Automatic Leave-One-Out (LOO)

When the training set has fewer samples than `n_splits`, the optimizer automatically switches to Leave-One-Out cross-validation. This avoids empty validation folds and provides the most data-efficient evaluation for very small datasets (common in materials optimisation with expensive experiments).

```python
# With only 8 samples and n_splits=10, LOO is used automatically
optimizer = GPCrossValidatedOptimizer(
    X_train=X_small,  # shape (8, 3)
    y_train=y_small,
    n_splits=10,      # Auto-switches to LOO (8 folds)
    random_state=42
)
```

## Customization

*   **Scoring**: `scoring="cv_rmse"`, `"nlpd"`, or `"combined"`.
*   **NLPD weight**: `nlpd_weight=0.5` (only for `"combined"` mode).
*   **Kernels**: `kernels=["RBF", "Matern52"]` to search only specific kernels.
*   **Lengthscale range**: `lengthscale_bounds=(0.05, 50.0)`.
*   **Kernel variance range**: `kernel_variance_bounds=(1e-4, 10.0)`.
*   **Noise variance range**: `noise_variance_bounds=(1e-6, 1.0)`.
*   **Full custom space**: `hyperopt_space={...}` for complete control.
*   **Cross-Validation**: `n_splits` and `random_state`.
*   **Hyperopt**: `max_evals` and `rstate_seed` in `optimize()`.

## Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please open an issue or submit a pull request to the GitHub repository: https://github.com/Shifa-Zhong/bayesian-gp-cvloss

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Shifa Zhong (sfzhong@tongji.edu.cn)
GitHub: [Shifa-Zhong](https://github.com/Shifa-Zhong)
