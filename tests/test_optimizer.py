"""
Tests for bayesian-gp-cvloss v0.2.0.

Covers:
  - Three scoring modes (cv_rmse, nlpd, combined)
  - Auto LOO for small samples
  - NLPD static method correctness
  - Backward compatibility with v0.1.x usage
  - Parameter validation
  - Trial result structure
"""
import pytest
import numpy as np
import pandas as pd

from bayesian_gp_cvloss import GPCrossValidatedOptimizer, DEFAULT_KERNELS, VALID_SCORING


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_data():
    """Medium-sized synthetic dataset for normal CV tests."""
    np.random.seed(42)
    n, d = 50, 3
    X = np.random.rand(n, d)
    y = np.sin(X[:, 0] * 2 * np.pi) + X[:, 1] ** 2 + np.random.randn(n) * 0.1
    return X, y


@pytest.fixture
def small_data():
    """Very small dataset to trigger LOO."""
    np.random.seed(7)
    n, d = 6, 2
    X = np.random.rand(n, d)
    y = X[:, 0] + np.random.randn(n) * 0.05
    return X, y


# ---------------------------------------------------------------------------
# 1. Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_invalid_scoring_rejected(self, synth_data):
        X, y = synth_data
        with pytest.raises(ValueError, match="scoring must be one of"):
            GPCrossValidatedOptimizer(X, y, scoring="bad")

    def test_nlpd_weight_out_of_range(self, synth_data):
        X, y = synth_data
        with pytest.raises(ValueError, match="nlpd_weight must be in"):
            GPCrossValidatedOptimizer(X, y, scoring="combined", nlpd_weight=0.0)
        with pytest.raises(ValueError, match="nlpd_weight must be in"):
            GPCrossValidatedOptimizer(X, y, scoring="combined", nlpd_weight=1.0)

    def test_valid_scoring_accepted(self, synth_data):
        X, y = synth_data
        for s in VALID_SCORING:
            kw = {"nlpd_weight": 0.5} if s == "combined" else {}
            opt = GPCrossValidatedOptimizer(X, y, scoring=s, **kw)
            assert opt.scoring == s

    def test_default_scoring_is_cv_rmse(self, synth_data):
        X, y = synth_data
        opt = GPCrossValidatedOptimizer(X, y)
        assert opt.scoring == "cv_rmse"


# ---------------------------------------------------------------------------
# 2. Auto LOO
# ---------------------------------------------------------------------------

class TestAutoLOO:
    def test_loo_triggered_for_small_data(self, small_data):
        X, y = small_data
        opt = GPCrossValidatedOptimizer(X, y, n_splits=10, random_state=42)
        assert opt._use_loo is True
        assert opt._effective_n_splits == X.shape[0]

    def test_kfold_used_for_sufficient_data(self, synth_data):
        X, y = synth_data
        opt = GPCrossValidatedOptimizer(X, y, n_splits=5, random_state=42)
        assert opt._use_loo is False
        assert opt._effective_n_splits == 5

    def test_loo_exact_boundary(self):
        """n_samples == n_splits should use KFold, not LOO."""
        np.random.seed(1)
        X = np.random.rand(5, 2)
        y = np.random.rand(5)
        opt = GPCrossValidatedOptimizer(X, y, n_splits=5, random_state=42)
        assert opt._use_loo is False


# ---------------------------------------------------------------------------
# 3. NLPD static method
# ---------------------------------------------------------------------------

class TestNLPD:
    def test_perfect_prediction_low_nlpd(self):
        """Perfect mean with tight variance should give low NLPD."""
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([1.0, 2.0, 3.0])
        var_tight = np.array([0.01, 0.01, 0.01])
        nlpd = GPCrossValidatedOptimizer._compute_nlpd(y, mu, var_tight)
        assert np.isfinite(nlpd)
        assert nlpd < 0  # log of small Gaussian → can be negative

    def test_bad_prediction_high_nlpd(self):
        """Wrong mean with tight variance should give high NLPD."""
        y = np.array([1.0, 2.0, 3.0])
        mu = np.array([10.0, 20.0, 30.0])
        var_tight = np.array([0.01, 0.01, 0.01])
        nlpd = GPCrossValidatedOptimizer._compute_nlpd(y, mu, var_tight)
        assert nlpd > 10  # very high penalty

    def test_overconfident_worse_than_calibrated(self):
        """Same error magnitude: narrow variance should score worse than wide."""
        y = np.array([1.0])
        mu = np.array([2.0])  # error = 1.0
        narrow_var = np.array([0.01])
        wide_var = np.array([10.0])
        nlpd_narrow = GPCrossValidatedOptimizer._compute_nlpd(y, mu, narrow_var)
        nlpd_wide = GPCrossValidatedOptimizer._compute_nlpd(y, mu, wide_var)
        assert nlpd_narrow > nlpd_wide, (
            "Overconfident (narrow var) should have worse NLPD than calibrated (wide var)"
        )

    def test_underconfident_worse_than_calibrated(self):
        """Near-perfect prediction: wide variance should score worse than tight."""
        y = np.array([1.0])
        mu = np.array([1.001])  # very small error
        tight_var = np.array([0.01])
        wide_var = np.array([100.0])
        nlpd_tight = GPCrossValidatedOptimizer._compute_nlpd(y, mu, tight_var)
        nlpd_wide = GPCrossValidatedOptimizer._compute_nlpd(y, mu, wide_var)
        assert nlpd_wide > nlpd_tight, (
            "Underconfident (wide var on good prediction) should be worse"
        )

    def test_zero_variance_clamped(self):
        """Zero variance should not cause NaN/Inf."""
        y = np.array([1.0])
        mu = np.array([1.0])
        zero_var = np.array([0.0])
        nlpd = GPCrossValidatedOptimizer._compute_nlpd(y, mu, zero_var)
        assert np.isfinite(nlpd)


# ---------------------------------------------------------------------------
# 4. Optimization runs (functional tests with small max_evals)
# ---------------------------------------------------------------------------

class TestOptimizationRuns:
    @pytest.mark.parametrize("scoring", ["cv_rmse", "nlpd", "combined"])
    def test_optimize_completes(self, synth_data, scoring):
        """Each scoring mode should complete without error."""
        X, y = synth_data
        kw = {"nlpd_weight": 0.5} if scoring == "combined" else {}
        opt = GPCrossValidatedOptimizer(
            X, y, scoring=scoring, n_splits=3, random_state=42, **kw
        )
        best = opt.optimize(max_evals=3)
        assert best is not None
        assert opt.best_model_ is not None

    @pytest.mark.parametrize("scoring", ["cv_rmse", "nlpd", "combined"])
    def test_trial_results_contain_all_metrics(self, synth_data, scoring):
        """Regardless of scoring, all trials should report cv_rmse and cv_nlpd."""
        X, y = synth_data
        kw = {"nlpd_weight": 0.5} if scoring == "combined" else {}
        opt = GPCrossValidatedOptimizer(
            X, y, scoring=scoring, n_splits=3, random_state=42, **kw
        )
        opt.optimize(max_evals=3)

        for trial in opt.trials.trials:
            r = trial['result']
            assert 'cv_rmse' in r, "Trial result missing cv_rmse"
            assert 'cv_nlpd' in r, "Trial result missing cv_nlpd"
            assert 'train_loss' in r, "Trial result missing train_loss"
            assert 'loss' in r

    def test_loo_optimization_completes(self, small_data):
        """LOO mode should complete optimisation on very small data."""
        X, y = small_data
        opt = GPCrossValidatedOptimizer(
            X, y, scoring="nlpd", n_splits=10, random_state=42
        )
        assert opt._use_loo is True
        best = opt.optimize(max_evals=3)
        assert best is not None


# ---------------------------------------------------------------------------
# 5. Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_v01x_usage_still_works(self, synth_data):
        """v0.1.x calling convention should still work identically."""
        X, y = synth_data
        opt = GPCrossValidatedOptimizer(
            X_train=X, y_train=y,
            hyperopt_space=None,
            n_splits=3, random_state=42
        )
        best = opt.optimize(max_evals=3)
        assert best is not None
        # loss should equal cv_rmse in default mode
        r = opt.trials.best_trial['result']
        assert abs(r['loss'] - r['cv_rmse']) < 1e-9

    def test_predict_works_after_optimize(self, synth_data):
        X, y = synth_data
        opt = GPCrossValidatedOptimizer(X, y, n_splits=3, random_state=42)
        opt.optimize(max_evals=3)
        mean, var = opt.predict(X[:5])
        assert mean is not None
        assert var is not None
        assert mean.shape[0] == 5
        assert var.shape[0] == 5

    def test_default_kernels_exported(self):
        assert "RBF" in DEFAULT_KERNELS
        assert "Matern52" in DEFAULT_KERNELS

    def test_valid_scoring_exported(self):
        assert VALID_SCORING == ("cv_rmse", "nlpd", "combined")


# ---------------------------------------------------------------------------
# 6. Prediction quality smoke test
# ---------------------------------------------------------------------------

class TestPredictionQuality:
    def test_predictions_on_original_scale(self, synth_data):
        """Predictions should be on original y scale, not centered."""
        X, y = synth_data
        opt = GPCrossValidatedOptimizer(X, y, n_splits=3, random_state=42)
        opt.optimize(max_evals=5)
        mean, _ = opt.predict(X)
        # Mean of predictions should be in the same ballpark as mean of y
        assert abs(np.mean(mean) - np.mean(y)) < 2 * np.std(y)

    def test_variance_positive(self, synth_data):
        X, y = synth_data
        opt = GPCrossValidatedOptimizer(X, y, n_splits=3, random_state=42)
        opt.optimize(max_evals=3)
        _, var = opt.predict(X)
        assert np.all(var > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
