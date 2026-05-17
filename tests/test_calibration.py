"""Tests for the ConformalCalibrator."""
import numpy as np
import pytest

from bayesian_gp_cvloss import GPCrossValidatedOptimizer, ConformalCalibrator


@pytest.fixture
def fitted_optimizer():
    """A GPCrossValidatedOptimizer fitted on cv_rmse mode."""
    rng = np.random.default_rng(0)
    n, d = 60, 2
    X = rng.uniform(0, 1, size=(n, d))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, 1] + rng.normal(0, 0.15, size=n)
    opt = GPCrossValidatedOptimizer(
        X, y, scoring="cv_rmse", n_splits=3, random_state=0
    )
    opt.optimize(max_evals=5)
    assert opt.best_model_ is not None
    return opt, X, y, rng


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_alpha_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalCalibrator(alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalCalibrator(alpha=1.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalCalibrator(alpha=-0.1)

    def test_alpha_in_range_accepted(self):
        for a in [0.01, 0.1, 0.5, 0.9, 0.99]:
            ConformalCalibrator(alpha=a)


# ---------------------------------------------------------------------------
# Fit-time validation
# ---------------------------------------------------------------------------

class TestFitValidation:
    def test_unfitted_optimizer_rejected(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, size=(20, 2))
        y = rng.normal(size=20)
        opt = GPCrossValidatedOptimizer(X, y, scoring="cv_rmse", random_state=0)
        # opt has best_model_=None at this point
        cal = ConformalCalibrator(alpha=0.1)
        with pytest.raises(RuntimeError, match="best_model_ is None"):
            cal.fit(opt, X, y)

    def test_calibration_too_small(self, fitted_optimizer):
        opt, _, _, rng = fitted_optimizer
        # alpha=0.1 needs n >= 9 for the (n+1) correction
        X_cal = rng.uniform(0, 1, size=(5, 2))
        y_cal = rng.normal(size=5)
        cal = ConformalCalibrator(alpha=0.1)
        with pytest.raises(ValueError, match="too small"):
            cal.fit(opt, X_cal, y_cal)

    def test_length_mismatch_rejected(self, fitted_optimizer):
        opt, _, _, rng = fitted_optimizer
        X_cal = rng.uniform(0, 1, size=(15, 2))
        y_cal = rng.normal(size=14)
        cal = ConformalCalibrator(alpha=0.1)
        with pytest.raises(ValueError, match="length mismatch"):
            cal.fit(opt, X_cal, y_cal)


# ---------------------------------------------------------------------------
# Quantile math: finite-sample (n+1) correction
# ---------------------------------------------------------------------------

class TestQuantileCorrection:
    """Verify q_ is the k-th smallest score where k = ceil((n+1)(1-alpha))."""

    def test_quantile_index_matches_formula(self, fitted_optimizer):
        opt, _, _, rng = fitted_optimizer
        n = 19
        X_cal = rng.uniform(0, 1, size=(n, 2))
        y_cal = np.sin(2 * np.pi * X_cal[:, 0]) + 0.5 * X_cal[:, 1]
        cal = ConformalCalibrator(alpha=0.1).fit(opt, X_cal, y_cal)

        # k = ceil(20 * 0.9) = 18
        expected_k = int(np.ceil((n + 1) * 0.9))
        assert expected_k == 18
        sorted_scores = np.sort(cal.scores_)
        assert cal.q_ == pytest.approx(sorted_scores[expected_k - 1])

    def test_q_is_positive(self, fitted_optimizer):
        opt, _, _, rng = fitted_optimizer
        X_cal = rng.uniform(0, 1, size=(30, 2))
        y_cal = rng.normal(size=30)
        cal = ConformalCalibrator(alpha=0.1).fit(opt, X_cal, y_cal)
        assert cal.q_ > 0

    def test_smaller_alpha_gives_larger_q(self, fitted_optimizer):
        """Tighter coverage requirement should yield a larger multiplier."""
        opt, _, _, rng = fitted_optimizer
        X_cal = rng.uniform(0, 1, size=(40, 2))
        y_cal = (
            np.sin(2 * np.pi * X_cal[:, 0]) + 0.5 * X_cal[:, 1]
            + rng.normal(0, 0.15, size=40)
        )
        cal_90 = ConformalCalibrator(alpha=0.1).fit(opt, X_cal, y_cal)
        cal_95 = ConformalCalibrator(alpha=0.05).fit(opt, X_cal, y_cal)
        assert cal_95.q_ >= cal_90.q_


# ---------------------------------------------------------------------------
# Predict interval API
# ---------------------------------------------------------------------------

class TestPredictInterval:
    def test_requires_fit(self, fitted_optimizer):
        cal = ConformalCalibrator(alpha=0.1)
        opt, X, _, _ = fitted_optimizer
        with pytest.raises(RuntimeError, match="Call fit"):
            cal.predict_interval(X[:5])

    def test_shapes(self, fitted_optimizer):
        opt, _, _, rng = fitted_optimizer
        X_cal = rng.uniform(0, 1, size=(25, 2))
        y_cal = rng.normal(size=25)
        cal = ConformalCalibrator(alpha=0.1).fit(opt, X_cal, y_cal)

        X_new = rng.uniform(0, 1, size=(7, 2))
        mean, lo, hi = cal.predict_interval(X_new)
        assert mean.shape == (7,)
        assert lo.shape == (7,)
        assert hi.shape == (7,)

    def test_intervals_ordered(self, fitted_optimizer):
        opt, _, _, rng = fitted_optimizer
        X_cal = rng.uniform(0, 1, size=(25, 2))
        y_cal = rng.normal(size=25)
        cal = ConformalCalibrator(alpha=0.1).fit(opt, X_cal, y_cal)

        X_new = rng.uniform(0, 1, size=(10, 2))
        mean, lo, hi = cal.predict_interval(X_new)
        assert np.all(lo <= mean)
        assert np.all(mean <= hi)

    def test_calibrated_half_width_positive(self, fitted_optimizer):
        opt, _, _, rng = fitted_optimizer
        X_cal = rng.uniform(0, 1, size=(25, 2))
        y_cal = rng.normal(size=25)
        cal = ConformalCalibrator(alpha=0.1).fit(opt, X_cal, y_cal)
        hw = cal.calibrated_half_width(rng.uniform(0, 1, size=(10, 2)))
        assert hw.shape == (10,)
        assert np.all(hw > 0)


# ---------------------------------------------------------------------------
# Empirical coverage on held-out data
# ---------------------------------------------------------------------------

class TestEmpiricalCoverage:
    """Marginal coverage on a fresh test set should be approximately >= 1-alpha.

    With finite samples the empirical coverage fluctuates, so we use a loose
    lower bound to keep the test deterministic-friendly.
    """

    def test_coverage_approximately_holds(self):
        rng = np.random.default_rng(123)
        n_train, n_cal, n_test = 50, 100, 200
        d = 2

        def f(X):
            return np.sin(2 * np.pi * X[:, 0]) + 0.5 * X[:, 1]

        X_train = rng.uniform(0, 1, size=(n_train, d))
        y_train = f(X_train) + rng.normal(0, 0.15, size=n_train)

        opt = GPCrossValidatedOptimizer(
            X_train, y_train, scoring="cv_rmse", n_splits=3, random_state=0
        )
        opt.optimize(max_evals=10)

        X_cal = rng.uniform(0, 1, size=(n_cal, d))
        y_cal = f(X_cal) + rng.normal(0, 0.15, size=n_cal)
        cal = ConformalCalibrator(alpha=0.1).fit(opt, X_cal, y_cal)

        X_test = rng.uniform(0, 1, size=(n_test, d))
        y_test = f(X_test) + rng.normal(0, 0.15, size=n_test)
        _, lo, hi = cal.predict_interval(X_test)
        covered = ((y_test >= lo) & (y_test <= hi)).mean()

        # Target is 0.9; allow generous slack for finite-sample variance.
        assert covered >= 0.80, f"Empirical coverage {covered:.3f} too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
