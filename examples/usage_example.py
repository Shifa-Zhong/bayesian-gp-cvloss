import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from hyperopt import hp # For defining a custom space, if desired

from bayesian_gp_cvloss import GPCrossValidatedOptimizer
# DEFAULT_KERNELS can still be imported if user wants to build a custom space with specific kernels
from bayesian_gp_cvloss.optimizer import DEFAULT_KERNELS

def create_sample_dataframe(num_samples=100, random_seed=42):
    np.random.seed(random_seed)
    data = {
        'cat_A': np.random.choice(['A1', 'A2', 'A3', 'A4'], size=num_samples),
        'cat_B': np.random.choice(['B1', 'B2', 'B3'], size=num_samples),
        'num_1': np.random.rand(num_samples) * 10,
        'num_2': np.random.randn(num_samples) * 5,
        'target': (np.random.rand(num_samples) * 10 +
                   np.random.randn(num_samples) * 1 +
                   2 * (np.random.choice([0,1,2], size=num_samples)))
    }
    df = pd.DataFrame(data)
    df.loc[df['cat_A'] == 'A1', 'target'] += 3
    df.loc[df['cat_A'] == 'A2', 'target'] -= 2
    df.loc[df['cat_B'] == 'B1', 'target'] += 1.5
    return df

if __name__ == "__main__":
    # 1. Create Sample Data
    raw_train_df = create_sample_dataframe(num_samples=150, random_seed=123)
    target_column = 'target'

    # --- User's Responsibility: Data Preprocessing ---
    print("Performing user-defined data preprocessing...")
    X_raw = raw_train_df.drop(columns=[target_column])
    y_raw = raw_train_df[target_column]

    categorical_features_manual = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features_manual = X_raw.select_dtypes(include=np.number).columns.tolist()

    from sklearn.preprocessing import OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_manual),
            ('cat', categorical_transformer, categorical_features_manual)
        ],
        remainder='passthrough'
    )

    X_processed_array = preprocessor.fit_transform(X_raw)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f'feat_{i}' for i in range(X_processed_array.shape[1])]

    X_train_processed = pd.DataFrame(X_processed_array, columns=feature_names, index=X_raw.index)
    y_train_processed = y_raw.copy()

    print(f"Shape of X_train_processed: {X_train_processed.shape}")
    print(f"Shape of y_train_processed: {y_train_processed.shape}")
    print("First 5 rows of X_train_processed:")
    print(X_train_processed.head())
    # --- End User Preprocessing ---

    # 2. Instantiate Optimizer
    # Option A: Use all defaults (data-dependent space auto-generated)
    optimizer = GPCrossValidatedOptimizer(
        X_train=X_train_processed,
        y_train=y_train_processed,
        n_splits=3,
        random_state=42
    )

    # Option B: Override specific settings
    # optimizer = GPCrossValidatedOptimizer(
    #     X_train=X_train_processed,
    #     y_train=y_train_processed,
    #     kernels=["RBF", "Matern52"],              # Only search these kernels
    #     lengthscale_bounds=(0.05, 50.0),           # Custom lengthscale range
    #     kernel_variance_bounds=(1e-4, 20.0),       # Custom kernel variance range
    #     noise_variance_bounds=(1e-6, 1.0),         # Custom noise variance range
    #     n_splits=3,
    #     random_state=42
    # )

    print(f"\nOptimizer initialized. Active kernels: {list(optimizer._active_kernels.keys())}")

    # 3. Run Optimization
    print("\nStarting optimization...")
    best_hyperparams = optimizer.optimize(max_evals=10)

    print(f"\nOptimization complete.")
    print(f"Best hyperparameters found: {optimizer.best_params}")

    trials = optimizer.get_optimization_results()
    if trials.best_trial and trials.best_trial['result']['status'] == 'ok':
         print(f"Best trial full params: {trials.best_trial['result']['params']}")
         print(f"Best trial loss (CV RMSE): {trials.best_trial['result']['loss']:.4f}")
         if 'train_loss' in trials.best_trial['result']:
             print(f"Best trial train_loss (CV Train RMSE): {trials.best_trial['result']['train_loss']:.4f}")
    else:
        print("No successful best trial found.")

    # 4. Make Predictions with the Best Model
    if optimizer.best_model_:
        print("\nRefitted best model acquired.")

        new_raw_data = create_sample_dataframe(num_samples=5, random_seed=789)
        actual_new_y = new_raw_data[target_column].copy()
        new_X_raw = new_raw_data.drop(columns=[target_column], errors='ignore')

        print(f"\nMaking predictions on {len(new_X_raw)} new samples...")

        # Must use the *same* preprocessor fitted on the training data
        X_new_processed_array = preprocessor.transform(new_X_raw)
        X_new_processed = pd.DataFrame(X_new_processed_array, columns=feature_names, index=new_X_raw.index)

        pred_mean, pred_var = optimizer.predict(X_new_processed)

        if pred_mean is not None:
            print(f"\nPredictions (mean): {pred_mean.flatten()}")
            print(f"Actual values: {actual_new_y.values}")
            from sklearn.metrics import mean_squared_error
            rmse_new_data = np.sqrt(mean_squared_error(actual_new_y, pred_mean))
            print(f"RMSE on new data: {rmse_new_data:.4f}")
        else:
            print("Prediction failed.")
    else:
        print("\nBest model was not refitted/available. Cannot make predictions.")

    print("\nExample script finished.")
