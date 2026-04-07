import pandas as pd
import numpy as np
import gpflow
from gpflow.utilities import set_trainable
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import logging

logger = logging.getLogger(__name__)

# Default GP Kernels from GPflow - user can specify these in the space
DEFAULT_KERNELS = {
    #"Matern12": gpflow.kernels.Matern12,
    "Matern32": gpflow.kernels.Matern32,
    "Matern52": gpflow.kernels.Matern52,
    "RBF": gpflow.kernels.RBF,
    "RationalQuadratic": gpflow.kernels.RationalQuadratic,
    #"Exponential": gpflow.kernels.Exponential
}

VALID_SCORING = ("cv_rmse", "nlpd", "combined")


class GPCrossValidatedOptimizer:
    """
    Optimizes hyperparameters for a Gaussian Process Regressor using Hyperopt
    with k-fold cross-validation.

    Supports three scoring objectives:

    * ``"cv_rmse"`` (default) – minimise mean cross-validated RMSE.
      Directly targets prediction accuracy; backward-compatible with v0.1.x.
    * ``"nlpd"``  – minimise mean cross-validated Negative Log Predictive
      Density.  Simultaneously rewards accurate means **and** well-calibrated
      uncertainty estimates, which is important when the GP model drives an
      acquisition function (e.g. in Bayesian optimisation).
    * ``"combined"`` – minimise a weighted sum
      ``nlpd_weight * NLPD + (1 - nlpd_weight) * RMSE`` (both
      min-max-normalised within each trial so the weight is meaningful).

    When the training set is very small (fewer samples than ``n_splits``), the
    splitter automatically falls back to Leave-One-Out (LOO) cross-validation
    to avoid folds with zero validation samples.

    Users can fine-tune individual search ranges via constructor kwargs
    (``kernels``, ``lengthscale_bounds``, ``kernel_variance_bounds``,
    ``noise_variance_bounds``) without having to build a full hyperopt space
    dict.  If ``hyperopt_space`` is given, it takes full precedence and all
    individual bound kwargs are ignored.
    """

    def __init__(self, X_train, y_train,
                 hyperopt_space=None,
                 kernels=None,
                 lengthscale_bounds=None,
                 kernel_variance_bounds=None,
                 noise_variance_bounds=None,
                 scoring="cv_rmse",
                 nlpd_weight=0.5,
                 n_splits=5, random_state=None):
        """
        Args:
            X_train (pd.DataFrame or np.ndarray): Preprocessed training features (2D).
            y_train (pd.Series or np.ndarray): Training target (1D or 2D single-column).
            hyperopt_space (dict, optional): Full Hyperopt search space.  If
                provided, all individual bound kwargs below are ignored.
            kernels (list of str, optional): Kernel names to search.  Must be
                keys in DEFAULT_KERNELS.  Defaults to all.
            lengthscale_bounds (tuple or None): (low, high) for lengthscale
                search range (uniform across features).  If None, per-feature
                data-dependent bounds are computed.
            kernel_variance_bounds (tuple or None): (low, high) for kernel
                variance.  If None, defaults to (1e-6, 2 * Var(y)).
            noise_variance_bounds (tuple or None): (low, high) for likelihood
                noise variance (log-uniform).  If None, data-dependent.
            scoring (str): Optimisation objective.  One of ``"cv_rmse"``,
                ``"nlpd"``, or ``"combined"``.
            nlpd_weight (float): Weight for the NLPD term when
                ``scoring="combined"``.  Must be in (0, 1).  Ignored for other
                scoring modes.
            n_splits (int): Number of folds for KFold CV.  If the training set
                has fewer samples, Leave-One-Out is used automatically.
            random_state (int, optional): Random seed for reproducibility.
        """
        # --- Validate X_train / y_train ---
        if not isinstance(X_train, (pd.DataFrame, np.ndarray)):
            raise ValueError("X_train must be a pandas DataFrame or NumPy ndarray.")
        if not isinstance(y_train, (pd.Series, np.ndarray)):
            raise ValueError("y_train must be a pandas Series or NumPy ndarray.")

        if isinstance(X_train, np.ndarray) and len(X_train.shape) != 2:
             raise ValueError("X_train as NumPy array must be 2D.")

        _y_data_internal = y_train.values if isinstance(y_train, pd.Series) else np.asarray(y_train)
        if len(_y_data_internal.shape) != 1 and not (len(_y_data_internal.shape) == 2 and _y_data_internal.shape[1] == 1):
            raise ValueError("y_train must be 1D or 2D with one column.")
        if len(_y_data_internal.shape) == 2:
            _y_data_internal = _y_data_internal.flatten()

        if X_train.shape[0] != _y_data_internal.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples.")

        self.X_train = X_train
        self.y_train_1d = _y_data_internal
        self.y_train_mean_ = np.mean(self.y_train_1d)
        self.n_splits = n_splits
        self.random_state = random_state
        self.num_features = X_train.shape[1]

        # --- Validate scoring ---
        if scoring not in VALID_SCORING:
            raise ValueError(
                f"scoring must be one of {VALID_SCORING}, got {scoring!r}"
            )
        self.scoring = scoring

        if scoring == "combined":
            if not (0 < nlpd_weight < 1):
                raise ValueError(
                    f"nlpd_weight must be in (0, 1), got {nlpd_weight}"
                )
        self.nlpd_weight = float(nlpd_weight)

        # --- Determine CV splitter (auto-LOO for small samples) ---
        n_samples = X_train.shape[0]
        if n_samples < n_splits:
            logger.info(
                f"n_samples ({n_samples}) < n_splits ({n_splits}): "
                f"automatically switching to Leave-One-Out cross-validation."
            )
            self._use_loo = True
            self._effective_n_splits = n_samples
        else:
            self._use_loo = False
            self._effective_n_splits = n_splits

        # --- Validate and store individual override kwargs ---
        # Active kernels
        if kernels is not None:
            if not isinstance(kernels, list) or len(kernels) == 0:
                raise ValueError("kernels must be a non-empty list of kernel name strings.")
            invalid = [k for k in kernels if k not in DEFAULT_KERNELS]
            if invalid:
                raise ValueError(
                    f"Unknown kernel(s): {invalid}. "
                    f"Valid names: {list(DEFAULT_KERNELS.keys())}"
                )
            self._active_kernels = {k: DEFAULT_KERNELS[k] for k in kernels}
        else:
            self._active_kernels = dict(DEFAULT_KERNELS)

        # Validate bound tuples
        self._lengthscale_bounds = self._validate_bounds(lengthscale_bounds, "lengthscale_bounds")
        self._kernel_variance_bounds = self._validate_bounds(kernel_variance_bounds, "kernel_variance_bounds")
        self._noise_variance_bounds = self._validate_bounds(noise_variance_bounds, "noise_variance_bounds")

        # Warn if individual overrides are provided alongside hyperopt_space
        if hyperopt_space is not None:
            overrides = {
                "kernels": kernels, "lengthscale_bounds": lengthscale_bounds,
                "kernel_variance_bounds": kernel_variance_bounds,
                "noise_variance_bounds": noise_variance_bounds,
            }
            provided = [k for k, v in overrides.items() if v is not None]
            if provided:
                logger.warning(
                    f"hyperopt_space is provided, so individual overrides "
                    f"{provided} will be ignored."
                )

        # Build hyperopt space
        if hyperopt_space is not None:
            self.hyperopt_space = hyperopt_space
        else:
            self.hyperopt_space = self._get_default_data_dependent_space()

        self.trials = Trials()
        self.best_params = None
        self.best_model_ = None
        self._iteration_count = 0

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_bounds(bounds, name):
        """Validate a (low, high) bounds tuple. Returns the tuple or None."""
        if bounds is None:
            return None
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(f"{name} must be a (low, high) tuple, got {bounds!r}")
        low, high = bounds
        if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
            raise ValueError(f"{name} values must be numeric, got ({type(low).__name__}, {type(high).__name__})")
        if low <= 0 or high <= 0:
            raise ValueError(f"{name} values must be positive, got ({low}, {high})")
        if low >= high:
            raise ValueError(f"{name} requires low < high, got ({low}, {high})")
        return (float(low), float(high))

    @staticmethod
    def _compute_nlpd(y_true, pred_mean, pred_var):
        """Compute mean Negative Log Predictive Density.

        NLPD(y, μ, σ²) = 0.5·log(2π) + 0.5·log(σ²) + 0.5·(y-μ)²/σ²

        Lower is better.  A well-calibrated model achieves a lower NLPD than
        one with the same RMSE but poorly-calibrated variance.
        """
        # Clamp variance to avoid log(0) and division by zero
        safe_var = np.maximum(pred_var, 1e-12)
        nlpd_per_point = (
            0.5 * np.log(2 * np.pi)
            + 0.5 * np.log(safe_var)
            + 0.5 * (y_true - pred_mean) ** 2 / safe_var
        )
        return float(np.mean(nlpd_per_point))

    # ------------------------------------------------------------------
    # Search space construction
    # ------------------------------------------------------------------

    def _compute_feature_lengthscale_bounds(self):
        """Compute data-driven per-feature lengthscale bounds based on feature std."""
        X_data = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
        bounds = []
        for i in range(self.num_features):
            std_i = np.std(X_data[:, i])
            if std_i < 1e-9:
                bounds.append((0.1, 100.0))
            else:
                low_i = max(1e-4, std_i * 0.01)
                high_i = max(low_i * 10, std_i * 100.0)
                bounds.append((low_i, high_i))
        return bounds

    def _get_default_data_dependent_space(self):
        """Generates a data-dependent default hyperparameter search space."""
        logger.info("Generating data-dependent default hyperparameter space.")

        # --- Lengthscales ---
        if self._lengthscale_bounds is None:
            per_feature_bounds = self._compute_feature_lengthscale_bounds()
            space = {
                f'lengthscales_{i}': hp.quniform(f'lengthscales_{i}', lo, hi, 0.01)
                for i, (lo, hi) in enumerate(per_feature_bounds)
            }
            logger.info(f"Auto lengthscale bounds per feature: {per_feature_bounds}")
        else:
            lo, hi = self._lengthscale_bounds
            space = {
                f'lengthscales_{i}': hp.quniform(f'lengthscales_{i}', lo, hi, 0.01)
                for i in range(self.num_features)
            }
            logger.info(f"User lengthscale bounds: ({lo}, {hi})")

        # --- Kernel variance ---
        y_var = np.var(self.y_train_1d)
        if self._kernel_variance_bounds is None:
            kernel_var_lower = 1e-6
            kernel_var_upper = float(2.0 * y_var)
            kernel_var_upper = max(kernel_var_lower * 10, kernel_var_upper)
        else:
            kernel_var_lower, kernel_var_upper = self._kernel_variance_bounds
        logger.info(f"Kernel variance range: ({kernel_var_lower:.2e}, {kernel_var_upper:.2e})")
        space['kernel_variance'] = hp.uniform('kernel_variance', kernel_var_lower, kernel_var_upper)

        # --- Noise variance ---
        y_std = np.std(self.y_train_1d)
        if self._noise_variance_bounds is None:
            if y_std > 1e-9:
                noise_var_lower_bound = (y_std / 100.0)**2
                noise_var_upper_bound = y_std**2
            else:
                noise_var_lower_bound = 1e-9
                noise_var_upper_bound = 1e-2
            noise_var_lower_bound = max(1e-9, float(noise_var_lower_bound))
            noise_var_upper_bound = max(noise_var_lower_bound * 1.1 + 1e-9, float(noise_var_upper_bound))
        else:
            noise_var_lower_bound, noise_var_upper_bound = self._noise_variance_bounds
        logger.info(f"Noise variance range: ({noise_var_lower_bound:.2e}, {noise_var_upper_bound:.2e})")
        space['likelihood_noise_variance'] = hp.loguniform(
            'likelihood_noise_variance',
            np.log(noise_var_lower_bound),
            np.log(noise_var_upper_bound)
        )

        # --- Kernel choice ---
        space['kernel_name'] = hp.choice('kernel_name', list(self._active_kernels.keys()))
        return space

    # ------------------------------------------------------------------
    # Core CV objective
    # ------------------------------------------------------------------

    def _objective(self, params):
        self._iteration_count += 1
        iteration_num = self._iteration_count

        # --- Build CV splitter ---
        if self._use_loo:
            splitter = LeaveOneOut()
        else:
            splitter = KFold(n_splits=self.n_splits, shuffle=True,
                             random_state=self.random_state)

        fold_rmses = []
        fold_train_rmses = []
        fold_nlpds = []

        kernel_name = params['kernel_name']
        selected_kernel_class = self._active_kernels.get(kernel_name) or DEFAULT_KERNELS.get(kernel_name)
        if selected_kernel_class is None:
            logger.error(f"Kernel '{kernel_name}' not found. Returning inf loss.")
            return {'loss': np.inf, 'status': STATUS_OK, 'params': params, 'iteration': iteration_num}

        kernel_hparams = {}
        try:
            lengthscales = np.array([params[f'lengthscales_{i}'] for i in range(self.num_features)], dtype=float)
        except KeyError as e:
            logger.error(f"Missing lengthscale param: {e}. Params: {params}")
            return {'loss': np.inf, 'status': STATUS_OK, 'params': params, 'iteration': iteration_num}

        if not lengthscales.shape[0] == self.num_features:
            logger.error(f"LS dim mismatch. Expected {self.num_features}, got {lengthscales.shape[0]}")
            return {'loss': np.inf, 'status': STATUS_OK, 'params': params, 'iteration': iteration_num}

        kernel_hparams['lengthscales'] = lengthscales
        kernel_hparams['variance'] = float(params['kernel_variance'])
        current_noise_variance = float(params['likelihood_noise_variance'])

        X_data_for_cv = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
        y_data_1d_for_cv = self.y_train_1d

        for fold_idx, (train_index, val_index) in enumerate(splitter.split(X_data_for_cv)):
            X_train_fold, X_val_fold = X_data_for_cv[train_index], X_data_for_cv[val_index]
            y_train_fold_1d, y_val_fold_1d = y_data_1d_for_cv[train_index], y_data_1d_for_cv[val_index]

            # Mean centering for this fold based on this fold's training y
            current_fold_y_train_mean = np.mean(y_train_fold_1d)
            y_train_fold_centered = y_train_fold_1d - current_fold_y_train_mean
            y_val_fold_centered = y_val_fold_1d - current_fold_y_train_mean

            y_train_fold_2d = y_train_fold_centered.reshape(-1, 1)

            try:
                fold_kernel = selected_kernel_class(**kernel_hparams)
                model = gpflow.models.GPR(data=(X_train_fold, y_train_fold_2d),
                                         kernel=fold_kernel,
                                         noise_variance=current_noise_variance)

                set_trainable(model.kernel.variance, False)
                set_trainable(model.kernel.lengthscales, False)
                set_trainable(model.likelihood.variance, False)

                y_pred_val_centered, y_pred_var_val = model.predict_y(X_val_fold)
                y_pred_train_centered, _ = model.predict_y(X_train_fold)

                y_pred_val_np = y_pred_val_centered.numpy().flatten()
                y_pred_var_np = y_pred_var_val.numpy().flatten()
                y_val_centered_flat = y_val_fold_centered.flatten()

                fold_rmse = np.sqrt(mean_squared_error(y_val_centered_flat, y_pred_val_np))
                fold_train_rmse = np.sqrt(mean_squared_error(
                    y_train_fold_centered, y_pred_train_centered.numpy().flatten()
                ))
                fold_nlpd = self._compute_nlpd(y_val_centered_flat, y_pred_val_np, y_pred_var_np)

                fold_rmses.append(fold_rmse)
                fold_train_rmses.append(fold_train_rmse)
                fold_nlpds.append(fold_nlpd)
            except Exception as e:
                logger.warning(f"Fold {fold_idx+1} error for params {params}: {e}. High loss.")
                fold_rmses.append(np.inf)
                fold_train_rmses.append(np.inf)
                fold_nlpds.append(np.inf)
                break

        # --- Aggregate fold metrics ---
        def _safe_mean(arr):
            return np.mean(arr) if arr and np.all(np.isfinite(arr)) else np.inf

        avg_cv_rmse = _safe_mean(fold_rmses)
        avg_train_rmse = _safe_mean(fold_train_rmses)
        avg_cv_nlpd = _safe_mean(fold_nlpds)

        # --- Select loss according to scoring mode ---
        if self.scoring == "cv_rmse":
            loss = avg_cv_rmse
        elif self.scoring == "nlpd":
            loss = avg_cv_nlpd
        else:  # combined
            # Min-max normalise both metrics within the trial so the weight
            # is meaningful regardless of value scales.
            # For the very first trial we don't have history, so we use both
            # raw values.  After a few trials the normalisation stabilises.
            if len(self.trials.trials) > 0:
                past_rmses = [
                    t['result'].get('cv_rmse', np.inf)
                    for t in self.trials.trials
                    if 'result' in t and t['result']['status'] == STATUS_OK
                ]
                past_nlpds = [
                    t['result'].get('cv_nlpd', np.inf)
                    for t in self.trials.trials
                    if 'result' in t and t['result']['status'] == STATUS_OK
                ]
                # Only normalise if we have finite history
                finite_rmses = [v for v in past_rmses if np.isfinite(v)]
                finite_nlpds = [v for v in past_nlpds if np.isfinite(v)]

                if finite_rmses and finite_nlpds:
                    rmse_min, rmse_max = min(finite_rmses), max(finite_rmses)
                    nlpd_min, nlpd_max = min(finite_nlpds), max(finite_nlpds)
                    rmse_range = rmse_max - rmse_min if rmse_max > rmse_min else 1.0
                    nlpd_range = nlpd_max - nlpd_min if nlpd_max > nlpd_min else 1.0
                    norm_rmse = (avg_cv_rmse - rmse_min) / rmse_range
                    norm_nlpd = (avg_cv_nlpd - nlpd_min) / nlpd_range
                    loss = (1 - self.nlpd_weight) * norm_rmse + self.nlpd_weight * norm_nlpd
                else:
                    loss = (1 - self.nlpd_weight) * avg_cv_rmse + self.nlpd_weight * avg_cv_nlpd
            else:
                loss = (1 - self.nlpd_weight) * avg_cv_rmse + self.nlpd_weight * avg_cv_nlpd

        ls_rounded = np.round(lengthscales, 2)
        cv_type = "LOO" if self._use_loo else f"{self._effective_n_splits}-Fold"
        logger.info(
            f"Iter: {iteration_num:>3} | {cv_type} RMSE: {avg_cv_rmse:<8.4f} | "
            f"NLPD: {avg_cv_nlpd:<8.4f} | Train RMSE: {avg_train_rmse:<8.4f} | "
            f"Loss({self.scoring}): {loss:<8.4f} | "
            f"Kernel: {kernel_name} | Var: {kernel_hparams['variance']:.4f} | "
            f"Noise: {current_noise_variance:.6f} | LS: {ls_rounded}"
        )

        return {
            'loss': loss,
            'status': STATUS_OK,
            'params': params,
            'iteration': iteration_num,
            'train_loss': avg_train_rmse,
            'cv_rmse': avg_cv_rmse,
            'cv_nlpd': avg_cv_nlpd,
        }

    # ------------------------------------------------------------------
    # Optimise / refit / predict
    # ------------------------------------------------------------------

    def optimize(self, max_evals=100, tpe_algo=tpe.suggest, early_stop_fn=None, rstate_seed=None):
        self._iteration_count = 0
        if rstate_seed is None and self.random_state is not None:
            rstate_seed = self.random_state

        rstate = np.random.default_rng(rstate_seed) if rstate_seed is not None else None

        self.best_params_raw_ = fmin(
            fn=self._objective,
            space=self.hyperopt_space,
            algo=tpe_algo,
            max_evals=max_evals,
            trials=self.trials,
            early_stop_fn=early_stop_fn,
            rstate=rstate
        )

        logger.info(f"Optimization finished. Best raw params from fmin: {self.best_params_raw_}")

        if self.trials.best_trial and 'result' in self.trials.best_trial and self.trials.best_trial['result']['status'] == STATUS_OK:
            self.best_params = self.trials.best_trial['result']['params']
            logger.info(f"Best full params from trials: {self.best_params}")
            best_result = self.trials.best_trial['result']
            logger.info(f"Best loss ({self.scoring}): {best_result['loss']:.4f}")
            logger.info(f"Best CV RMSE: {best_result['cv_rmse']:.4f}")
            logger.info(f"Best CV NLPD: {best_result['cv_nlpd']:.4f}")
            logger.info(f"Best Train RMSE: {best_result['train_loss']:.4f}")
            self.refit_best_model()
        else:
            self.best_params = None
            logger.warning("Optimization did not yield a valid best trial. Model not refitted.")

        return self.best_params

    def refit_best_model(self):
        if not self.best_params:
            logger.warning("No valid best parameters. Cannot refit model.")
            self.best_model_ = None
            return None

        params_for_refit = self.best_params
        kernel_name = params_for_refit['kernel_name']
        selected_kernel_class = self._active_kernels.get(kernel_name) or DEFAULT_KERNELS.get(kernel_name)

        if selected_kernel_class is None:
            logger.error(f"Kernel '{kernel_name}' not found. Cannot refit.")
            self.best_model_ = None
            return None

        kernel_hparams = {}
        lengthscales = np.array([params_for_refit[f'lengthscales_{i}'] for i in range(self.num_features)], dtype=float)
        kernel_hparams['lengthscales'] = lengthscales
        kernel_hparams['variance'] = float(params_for_refit['kernel_variance'])
        noise_var_refit = float(params_for_refit['likelihood_noise_variance'])

        X_data_refit = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
        y_train_centered_for_refit = (self.y_train_1d - self.y_train_mean_).reshape(-1,1)

        try:
            best_kernel = selected_kernel_class(**kernel_hparams)
            self.best_model_ = gpflow.models.GPR(
                data=(X_data_refit, y_train_centered_for_refit),
                kernel=best_kernel,
                noise_variance=noise_var_refit
            )
            set_trainable(self.best_model_.kernel.variance, False)
            set_trainable(self.best_model_.kernel.lengthscales, False)
            set_trainable(self.best_model_.likelihood.variance, False)
            logger.info(f"Successfully refitted GPR model: {params_for_refit}")
        except Exception as e:
            logger.error(f"Error refitting model with params {params_for_refit}: {e}")
            self.best_model_ = None
        return self.best_model_

    def predict(self, X_new_processed):
        if self.best_model_ is None:
            logger.error("No best model. Run optimize() and ensure refit was successful.")
            return None, None
        if not isinstance(X_new_processed, (pd.DataFrame, np.ndarray)):
            raise ValueError("X_new_processed must be pd.DataFrame or np.ndarray.")

        if X_new_processed.shape[1] != self.num_features:
            raise ValueError(f"X_new has {X_new_processed.shape[1]} features, model expects {self.num_features}.")

        X_new_values = X_new_processed.values if isinstance(X_new_processed, pd.DataFrame) else X_new_processed

        try:
            pred_mean_centered, pred_var = self.best_model_.predict_y(X_new_values)
            pred_mean_original_scale = pred_mean_centered.numpy() + self.y_train_mean_
            return pred_mean_original_scale, pred_var.numpy()
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None, None

    def get_optimization_results(self):
        return self.trials
