"""Split conformal prediction calibrator for a fitted GP.

Pairs with ``GPCrossValidatedOptimizer(scoring="cv_rmse")``: let the GP
hyperparameter search target mean accuracy (RMSE), then wrap the trained
model with this calibrator to get prediction intervals with a finite-sample
marginal coverage guarantee.

The score function is locally adaptive::

    s_i = |y_i - mu_i| / sigma_i

so the calibrator preserves the GP's per-point uncertainty *ordering* and
only fixes the overall scale. This is preferable to the absolute-residual
score when the GP's relative uncertainties (high-variance points really are
the harder points) are trustworthy.

Two entry points
----------------
* ``fit(optimizer, X_cal, y_cal)`` -- classic split conformal. Requires a
  held-out calibration set disjoint from training data. Exact 1-alpha
  marginal coverage under exchangeability.
* ``fit_cv(optimizer, n_splits=None)`` -- naive cross-conformal. Re-runs
  K-fold CV with the optimizer's already-selected best hyperparameters
  (so the OOF residuals are unbiased given the hparams) and uses those as
  calibration scores. No held-out set needed. Coverage is approximately
  1-alpha (the formal split-CP proof does not apply because the calibration
  models are fold-specific while predictions at test time use the global
  model trained on all data), but in practice it tracks nominal coverage
  closely on exchangeable data.

Coverage guarantee
------------------
For ``fit``: under exchangeability of calibration and test points, for any
miscoverage level ``alpha in (0, 1)`` and calibration set of size ``n``::

    P( y* in [mu* - q*sigma*, mu* + q*sigma*] )  >=  1 - alpha

where ``q`` is the ``ceil((n+1)(1-alpha))``-th smallest calibration score.
The ``(n+1)`` term is the finite-sample correction; omitting it makes the
guarantee only asymptotic.
"""
import logging

import gpflow
import numpy as np
import pandas as pd
from gpflow.utilities import set_trainable
from sklearn.model_selection import KFold, LeaveOneOut

logger = logging.getLogger(__name__)


class ConformalCalibrator:
    """Locally adaptive split conformal prediction on top of a fitted GP.

    Parameters
    ----------
    alpha : float
        Target miscoverage level in (0, 1). ``alpha=0.1`` yields 90% intervals.

    Attributes (after ``fit``)
    --------------------------
    q_ : float
        Multiplier applied to GP sigma to form the prediction interval radius.
    scores_ : ndarray of shape (n_cal,)
        Per-point calibration scores ``|y - mu| / sigma``.
    n_cal_ : int
        Calibration set size.
    """

    def __init__(self, alpha=0.1):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self._optimizer = None

    def fit(self, optimizer, X_cal, y_cal):
        """Compute the conformal quantile from a held-out calibration set.

        ``optimizer`` must be a fitted ``GPCrossValidatedOptimizer`` (i.e.
        ``optimizer.best_model_`` is not None). ``X_cal`` and ``y_cal`` MUST
        be disjoint from the optimizer's training data, otherwise the
        exchangeability assumption that underlies the coverage guarantee is
        violated and the resulting intervals will be over-optimistic.
        """
        if getattr(optimizer, "best_model_", None) is None:
            raise RuntimeError(
                "optimizer.best_model_ is None. Call optimizer.optimize() first."
            )

        mean, var = optimizer.predict(X_cal)
        if mean is None or var is None:
            raise RuntimeError("optimizer.predict() returned None.")

        mean = np.asarray(mean).flatten()
        var = np.asarray(var).flatten()
        y_cal = np.asarray(y_cal).flatten()

        if mean.shape[0] != y_cal.shape[0]:
            raise ValueError(
                f"X_cal and y_cal length mismatch: "
                f"got {mean.shape[0]} predictions vs {y_cal.shape[0]} targets."
            )

        n = y_cal.shape[0]
        min_n = int(np.ceil(1.0 / self.alpha)) - 1
        if n < min_n + 1:
            raise ValueError(
                f"Calibration set too small: n={n}, but alpha={self.alpha} "
                f"requires n >= {min_n + 1} for the (n+1) quantile to exist."
            )

        sigma = np.sqrt(np.maximum(var, 1e-12))
        scores = np.abs(y_cal - mean) / sigma

        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        if k > n:
            raise ValueError(
                f"Cannot compute conformal quantile: k={k} > n={n}. "
                f"Increase calibration set size."
            )

        sorted_scores = np.sort(scores)
        self.q_ = float(sorted_scores[k - 1])
        self.scores_ = scores
        self.n_cal_ = n
        self._optimizer = optimizer
        return self

    def fit_cv(self, optimizer, n_splits=None, random_state=None):
        """Calibrate using OOF residuals from a fresh CV run with best hparams.

        Re-runs K-fold (or LOO, for small samples) cross-validation on the
        optimizer's training data using the already-selected best
        hyperparameters, then uses the resulting out-of-fold residuals as
        conformal calibration scores. No separate held-out set is required.

        Parameters
        ----------
        optimizer : GPCrossValidatedOptimizer
            A fitted optimizer (``optimize()`` already called).
        n_splits : int, optional
            Number of folds. Defaults to ``optimizer.n_splits``. Auto-falls
            back to Leave-One-Out if ``n_samples < n_splits``.
        random_state : int, optional
            Random seed for the KFold splitter. Defaults to
            ``optimizer.random_state``.

        Notes
        -----
        IMPORTANT: do NOT reuse residuals collected DURING hyperopt's search
        -- those are post-selection biased (hyperopt picked the trial with
        the smallest CV-RMSE, so its residuals systematically underestimate
        generalisation error). This method re-runs CV AFTER hyperparameters
        are frozen, which removes that bias.

        Coverage is approximately ``1 - alpha`` (naive cross-conformal): the
        formal split-CP guarantee is weakened because the calibration scores
        come from fold-specific models while test-time predictions use the
        global model trained on all data. Empirical coverage typically lands
        within a few percent of nominal on exchangeable data.
        """
        if getattr(optimizer, "best_params", None) is None or \
                getattr(optimizer, "best_model_", None) is None:
            raise RuntimeError(
                "optimizer is not fitted. Call optimizer.optimize() first."
            )

        X = optimizer.X_train.values if isinstance(optimizer.X_train, pd.DataFrame) \
            else optimizer.X_train
        y = optimizer.y_train_1d
        n = y.shape[0]

        n_splits_eff = n_splits if n_splits is not None else optimizer.n_splits
        rs = random_state if random_state is not None else optimizer.random_state

        if n < n_splits_eff:
            splitter = LeaveOneOut()
            logger.info(
                f"fit_cv: n={n} < n_splits={n_splits_eff}, using Leave-One-Out."
            )
        else:
            splitter = KFold(n_splits=n_splits_eff, shuffle=True, random_state=rs)

        min_n = int(np.ceil(1.0 / self.alpha)) - 1
        if n < min_n + 1:
            raise ValueError(
                f"Training set too small for alpha={self.alpha}: n={n}, "
                f"need n >= {min_n + 1} for the (n+1) quantile to exist."
            )

        bp = optimizer.best_params
        kernel_name = bp['kernel_name']
        kernel_cls = optimizer._active_kernels.get(kernel_name)
        if kernel_cls is None:
            raise RuntimeError(
                f"Kernel '{kernel_name}' from best_params not found in "
                f"optimizer._active_kernels."
            )
        n_features = X.shape[1]
        lengthscales = np.array(
            [bp[f'lengthscales_{i}'] for i in range(n_features)], dtype=float
        )
        kernel_variance = float(bp['kernel_variance'])
        noise_variance = float(bp['likelihood_noise_variance'])

        scores = np.full(n, np.nan)
        for train_idx, val_idx in splitter.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            fold_mean = float(np.mean(y_tr))
            y_tr_c = (y_tr - fold_mean).reshape(-1, 1)

            kernel = kernel_cls(lengthscales=lengthscales, variance=kernel_variance)
            model = gpflow.models.GPR(
                data=(X_tr, y_tr_c),
                kernel=kernel,
                noise_variance=noise_variance,
            )
            set_trainable(model.kernel.variance, False)
            set_trainable(model.kernel.lengthscales, False)
            set_trainable(model.likelihood.variance, False)

            mu_c, var = model.predict_y(X_val)
            mu_c = mu_c.numpy().flatten()
            var = var.numpy().flatten()

            y_val_c = y_val - fold_mean
            sigma = np.sqrt(np.maximum(var, 1e-12))
            scores[val_idx] = np.abs(y_val_c - mu_c) / sigma

        if np.any(np.isnan(scores)):
            missing = int(np.sum(np.isnan(scores)))
            raise RuntimeError(
                f"fit_cv: {missing} training points were not assigned to any "
                f"validation fold; CV splitter produced incomplete coverage."
            )

        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        if k > n:
            raise ValueError(
                f"Cannot compute conformal quantile: k={k} > n={n}. "
                f"Increase training set size or alpha."
            )

        sorted_scores = np.sort(scores)
        self.q_ = float(sorted_scores[k - 1])
        self.scores_ = scores
        self.n_cal_ = n
        self._optimizer = optimizer
        return self

    def predict_interval(self, X_new):
        """Return ``(mean, lower, upper)`` for each row of ``X_new``.

        Each interval has marginal coverage ``>= 1 - alpha`` under
        exchangeability with the calibration set.
        """
        self._check_fitted()
        mean, var = self._optimizer.predict(X_new)
        mean = np.asarray(mean).flatten()
        sigma = np.sqrt(np.maximum(np.asarray(var).flatten(), 1e-12))
        half_width = self.q_ * sigma
        return mean, mean - half_width, mean + half_width

    def calibrated_half_width(self, X_new):
        """Return the per-point interval half-width ``q_ * sigma`` for ``X_new``.

        Useful when you want to plug a "calibrated uncertainty" into an
        external workflow. Note this is a conformal quantile, not a Gaussian
        standard deviation -- do not pretend it is sigma in a normal CDF.
        """
        self._check_fitted()
        _, var = self._optimizer.predict(X_new)
        sigma = np.sqrt(np.maximum(np.asarray(var).flatten(), 1e-12))
        return self.q_ * sigma

    def _check_fitted(self):
        if self._optimizer is None or not hasattr(self, "q_"):
            raise RuntimeError("Call fit() before predict_interval().")
