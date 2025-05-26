import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from hyperopt import hp # For defining a custom space, if desired

from gpr_hyperopt_cv import GPCrossValidatedOptimizer
# DEFAULT_KERNELS can still be imported if user wants to build a custom space with specific kernels
from gpr_hyperopt_cv.optimizer import DEFAULT_KERNELS

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
    # The user must prepare their X and y data to be purely numerical and scaled appropriately.
    print("Performing user-defined data preprocessing...")
    X_raw = raw_train_df.drop(columns=[target_column])
    y_raw = raw_train_df[target_column]

    # Example: Using sklearn ColumnTransformer for one-hot encoding and scaling
    # Identify categorical and numerical features from X_raw
    categorical_features_manual = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features_manual = X_raw.select_dtypes(include=np.number).columns.tolist()

    # Create transformers
    # For simplicity, using OneHotEncoder. For high cardinality, other encoders might be better.
    # The GPCrossValidatedOptimizer does not care how features are generated, only that they are numeric.
    from sklearn.preprocessing import OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features_manual),
            ('cat', categorical_transformer, categorical_features_manual)
        ],
        remainder='passthrough' # Pass through any columns not specified (should be none if lists are correct)
    )

    # Fit the preprocessor on the raw training features and transform it
    X_processed_array = preprocessor.fit_transform(X_raw)
    # Convert back to DataFrame to see feature names (optional, as optimizer takes array)
    # Getting feature names after one-hot encoding can be tricky, this is a simplified way
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError: # Older sklearn
        # Manual construction or just use array without names
        feature_names = [f'feat_{i}' for i in range(X_processed_array.shape[1])]
        
    X_train_processed = pd.DataFrame(X_processed_array, columns=feature_names, index=X_raw.index)
    y_train_processed = y_raw.copy() # Already a Series, ensure it's 1D

    print(f"Shape of X_train_processed: {X_train_processed.shape}")
    print(f"Shape of y_train_processed: {y_train_processed.shape}")
    print("First 5 rows of X_train_processed:")
    print(X_train_processed.head())
    # --- End User Preprocessing ---

    # 2. Define Hyperparameter Search Space (Optional)
    # If `hyperopt_space` is None when creating GPCrossValidatedOptimizer, 
    # it will generate a data-dependent default space.
    # Here, we explicitly show how a user *could* define one.
    
    # num_processed_features = X_processed.shape[1] # Already have this
    # custom_space = {f'lengthscales_{i}': hp.quniform(f'lengthscales_{i}', 0.1, 100, 0.1) for i in range(num_processed_features)}
    # custom_space.update({
    #     'kernel_variance': hp.uniform('kernel_variance', 1e-3, float(y_processed.var()) + 1e-6),
    #     'likelihood_noise_variance': hp.loguniform('likelihood_noise_variance', 
    #                                                 np.log(max(1e-9, (y_processed.std()/100)**2)), 
    #                                                 np.log(max(1e-8, (y_processed.std()/2)**2))),
    #     'kernel_name': hp.choice('kernel_name', ['RBF', 'Matern52'])
    # })
    # For this example, we will let the optimizer create its default space.
    custom_space = None 
    print("\nUsing optimizer's data-dependent default hyperparameter space.")

    # 3. Instantiate Optimizer with preprocessed data
    optimizer = GPCrossValidatedOptimizer(
        X_train=X_train_processed, 
        y_train=y_train_processed, 
        hyperopt_space=custom_space,  # Can be None to use data-dependent default
        n_splits=3, 
        random_state=42
    )
    print(f"Optimizer initialized. Using space: {optimizer.hyperopt_space}")

    # 4. Run Optimization
    print("\nStarting optimization...")
    best_hyperparams = optimizer.optimize(max_evals=10) # Small max_evals for quick example

    print(f"\nOptimization complete.")
    print(f"Best hyperparameters found (from optimizer.best_params): {optimizer.best_params}")
    
    trials = optimizer.get_optimization_results()
    if trials.best_trial and trials.best_trial['result']['status'] == 'ok':
         print(f"Best trial full params: {trials.best_trial['result']['params']}")
         print(f"Best trial loss (CV RMSE): {trials.best_trial['result']['loss']:.4f}")
         if 'train_loss' in trials.best_trial['result']:
             print(f"Best trial train_loss (CV Train RMSE): {trials.best_trial['result']['train_loss']:.4f}")
    else:
        print("No successful best trial found.")

    # 5. Make Predictions with the Best Model
    if optimizer.best_model_:
        print("\nRefitted best model acquired.")
        
        new_raw_data = create_sample_dataframe(num_samples=5, random_seed=789)
        actual_new_y = new_raw_data[target_column].copy()
        new_X_raw = new_raw_data.drop(columns=[target_column], errors='ignore')
        
        print(f"\nMaking predictions on {len(new_X_raw)} new samples...")
        print("New raw data for prediction (first few rows):")
        print(new_X_raw.head())

        # --- User Preprocessing for New Data ---
        # Must use the *same* preprocessor fitted on the training data
        X_new_processed_array = preprocessor.transform(new_X_raw)
        X_new_processed = pd.DataFrame(X_new_processed_array, columns=feature_names, index=new_X_raw.index)
        # --- End User Preprocessing for New Data ---
        
        print("\nNew processed data for prediction (first few rows):")
        print(X_new_processed.head())

        pred_mean, pred_var = optimizer.predict(X_new_processed)

        if pred_mean is not None:
            print("\nPredictions (mean):")
            print(pred_mean.flatten())
            
            print("Actual values for new data:")
            print(actual_new_y.values)
            from sklearn.metrics import mean_squared_error
            rmse_new_data = np.sqrt(mean_squared_error(actual_new_y, pred_mean))
            print(f"RMSE on new data: {rmse_new_data:.4f}")
        else:
            print("Prediction failed.")
    else:
        print("\nBest model was not refitted/available. Cannot make predictions.")

    print("\nExample script finished.") 