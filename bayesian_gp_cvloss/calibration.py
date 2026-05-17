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

Coverage guarantee
------------------
Under exchangeability of calibration and test points, for any miscoverage
level ``alpha in (0, 1)`` and calibration set of size ``n``::

    P( y* in [mu* - q*sigma*, mu* + q*sigma*] )  >=  1 - alpha

where ``q`` is the ``ceil((n+1)(1-alpha))``-th smallest calibration score.
The ``(n+1)`` term is the finite-sample correction; omitting it makes the
guarantee only asymptotic.
"""
import numpy as np


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
