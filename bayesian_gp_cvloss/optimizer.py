import pandas as pd
import numpy as np
import gpflow
from gpflow.utilities import set_trainable
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
# StandardScaler and JamesSteinEncoder are no longer direct dependencies for the class
# but might be used by the user or the optional preprocessing utility.
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import tensorflow as tf
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

class GPCrossValidatedOptimizer:
    """
    Optimizes hyperparameters for a Gaussian Process Regressor using Hyperopt
    with k-fold cross-validation, minimizing RMSE. 
    Assumes input data X_train and y_train are already preprocessed (numerical and scaled).
    Generates a data-dependent default hyperparameter space if none is provided.
    """
    def __init__(self, X_train, y_train, 
                 hyperopt_space=None, n_splits=5, random_state=None):
        """
        Args:
            X_train (pd.DataFrame or np.ndarray): The preprocessed training feature dataset.
            y_train (pd.Series or np.ndarray): The preprocessed training target variable.
            hyperopt_space (dict, optional): Hyperopt search space. If None, a data-dependent default space is generated.
            n_splits (int): Number of folds for KFold cross-validation.
            random_state (int, optional): Random seed for KFold and Hyperopt for reproducibility.
        """
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
        self.y_train_1d = _y_data_internal # Store as 1D for var/std calculations and mean subtraction reference
        self.y_train_mean_ = np.mean(self.y_train_1d) # Store the mean of the original y_train
        self.n_splits = n_splits
        self.random_state = random_state
        
        self.num_features = X_train.shape[1]
        self.hyperopt_space = hyperopt_space if hyperopt_space is not None else self._get_default_data_dependent_space()
        self.trials = Trials()
        self.best_params = None
        self.best_model_ = None
        self._iteration_count = 0

    def _get_default_data_dependent_space(self):
        """Generates a data-dependent default hyperparameter search space."""
        logger.info("Generating data-dependent default hyperparameter space.")
        # Using y_train_1d which is y_train before any internal mean centering for GP models
        space = {f'lengthscales_{i}': hp.quniform(f'lengthscales_{i}', 0.1, 100, 0.01) for i in range(self.num_features)}
        
        y_var = np.var(self.y_train_1d) 
        y_std = np.std(self.y_train_1d)

        kernel_var_upper = float(y_var)
        kernel_var_lower = 0 
        logger.info(f"Default kernel_variance range: ({kernel_var_lower:.2e}, {kernel_var_upper:.2e})")
        space['kernel_variance'] = hp.uniform('kernel_variance', kernel_var_lower, kernel_var_upper)

        if y_std > 1e-9:
            noise_var_lower_bound = (y_std / 100.0)**2
            noise_var_upper_bound = (y_std / 2.0)**2
        else:
            noise_var_lower_bound = 1e-9 
            noise_var_upper_bound = 1e-2
        
        noise_var_lower_bound = max(1e-9, float(noise_var_lower_bound))
        noise_var_upper_bound = max(noise_var_lower_bound * 1.1 + 1e-9, float(noise_var_upper_bound))

        logger.info(f"Default likelihood_noise_variance loguniform range (effective): ({noise_var_lower_bound:.2e}, {noise_var_upper_bound:.2e})")
        space['likelihood_noise_variance'] = hp.loguniform(
            'likelihood_noise_variance', 
            np.log(noise_var_lower_bound), 
            np.log(noise_var_upper_bound)
        )
        
        space['kernel_name'] = hp.choice('kernel_name', list(DEFAULT_KERNELS.keys()))
        return space

    def _objective(self, params):
        self._iteration_count += 1
        iteration_num = self._iteration_count

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_rmses = []
        fold_train_rmses = []
        
        kernel_name = params['kernel_name']
        selected_kernel_class = DEFAULT_KERNELS[kernel_name]
        
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

        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_data_for_cv)):
            X_train_fold, X_val_fold = X_data_for_cv[train_index], X_data_for_cv[val_index]
            y_train_fold_1d, y_val_fold_1d = y_data_1d_for_cv[train_index], y_data_1d_for_cv[val_index]
            
            # Mean centering for this fold based on this fold's training y
            current_fold_y_train_mean = np.mean(y_train_fold_1d)
            y_train_fold_centered = y_train_fold_1d - current_fold_y_train_mean
            y_val_fold_centered = y_val_fold_1d - current_fold_y_train_mean
            
            y_train_fold_2d = y_train_fold_centered.reshape(-1,1)
            y_val_fold_2d = y_val_fold_centered.reshape(-1,1) # This is y_val - mean(y_train_fold)

            try:
                fold_kernel = selected_kernel_class(**kernel_hparams)
                model = gpflow.models.GPR(data=(X_train_fold, y_train_fold_2d), 
                                         kernel=fold_kernel, 
                                         noise_variance=current_noise_variance)
                
                set_trainable(model.kernel.variance, False)
                set_trainable(model.kernel.lengthscales, False)
                set_trainable(model.likelihood.variance, False)

                # Predictions are on the centered scale
                y_pred_val_centered, _ = model.predict_y(X_val_fold)
                y_pred_train_centered, _ = model.predict_y(X_train_fold)
                
                # RMSE is calculated on centered y_val and centered predictions
                # OR add mean back to predictions and compare with original y_val_fold_1d
                # The current notebook code subtracts fold_train_mean from y_val_fold_2d for training GP model, 
                # so predictions are already centered around that. So RMSE on y_val_fold_centered is correct.
                fold_rmse = np.sqrt(mean_squared_error(y_val_fold_centered, y_pred_val_centered.numpy()))
                fold_train_rmse = np.sqrt(mean_squared_error(y_train_fold_centered, y_pred_train_centered.numpy()))
                fold_rmses.append(fold_rmse)
                fold_train_rmses.append(fold_train_rmse)
            except Exception as e:
                logger.warning(f"Fold {fold_idx+1} error for params {params}: {e}. High loss.")
                fold_rmses.append(np.inf)
                fold_train_rmses.append(np.inf)
                break 
        
        avg_cv_rmse = np.mean(fold_rmses) if fold_rmses and np.all(np.isfinite(fold_rmses)) else np.inf
        avg_train_rmse = np.mean(fold_train_rmses) if fold_train_rmses and np.all(np.isfinite(fold_train_rmses)) else np.inf
        
        ls_rounded = np.round(lengthscales,2)
        logger.info(f"Iter: {iteration_num:>3} | CV RMSE: {avg_cv_rmse:<8.4f} | Train RMSE: {avg_train_rmse:<8.4f} | Kernel: {kernel_name} | Var: {kernel_hparams['variance']:.4f} | Noise: {current_noise_variance:.6f} | LS: {ls_rounded}")
        
        return {
            'loss': avg_cv_rmse, 
            'status': STATUS_OK, 
            'params': params, 
            'iteration': iteration_num,
            'train_loss': avg_train_rmse
        }

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
            logger.info(f"Best CV RMSE from trials: {self.trials.best_trial['result']['loss']}")
            logger.info(f"Best CV Train RMSE from trials: {self.trials.best_trial['result']['train_loss']}")
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
        selected_kernel_class = DEFAULT_KERNELS.get(kernel_name)

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
        # For refitting, use the original y_train_1d and subtract its overall mean (self.y_train_mean_)
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
            # Add back the overall mean of the original y_train used for refitting
            pred_mean_original_scale = pred_mean_centered.numpy() + self.y_train_mean_ 
            return pred_mean_original_scale, pred_var.numpy()
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None, None

    def get_optimization_results(self):
        return self.trials

# End of GPCrossValidatedOptimizer class 