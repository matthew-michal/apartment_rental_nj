import pandas as pd
import numpy as np
import boto3
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import mlflow
import mlflow.xgboost
import pickle
from datetime import datetime
from prefect import flow, task
import os
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder

from feature_engineering import find_nearest_stations, encode_categorical_variables

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET")
TRAINING_DATA_BUCKET = os.getenv("TRAINING_DATA_BUCKET")
RUN_ID_FILE = "run_id.txt"
MODEL_PREFIX = "models/"
REFERENCE_DATA_PREFIX = "reference_data/"
TRAINING_DATA_PREFIX = "training_data/"

@task(log_prints=True)
def load_training_data() -> pd.DataFrame:
    """
    Load all training data from S3 and combine into single dataset.
    """
    try:
        s3_client = boto3.client('s3')
        
        # List all training data files
        response = s3_client.list_objects_v2(
            Bucket=TRAINING_DATA_BUCKET,
            Prefix=TRAINING_DATA_PREFIX
        )
        
        if 'Contents' not in response:
            raise ValueError("No training data found in S3")
        
        # Load and combine all CSV files
        dataframes = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.csv'):
                csv_obj = s3_client.get_object(Bucket=TRAINING_DATA_BUCKET, Key=obj['Key'])
                df = pd.read_csv(csv_obj['Body'])
                dataframes.append(df)
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Remove duplicates and sort by date
        combined_df = combined_df.drop_duplicates()
        if 'prediction_date' in combined_df.columns:
            combined_df = combined_df.sort_values('prediction_date')
        
        print(f"Loaded {len(combined_df)} training samples from {len(dataframes)} files")
        return combined_df
        
    except Exception as e:
        print(f"Error loading training data: {e}")
        raise

@task(log_prints=True)
def prepare_training_data(df: pd.DataFrame) -> tuple:
    """
    Prepare training data by performing feature engineering and fitting encoders.
    """
    try:
        from feature_engineering import perform_feature_engineering_training
        
        # Perform feature engineering and get encoders
        engineered_data, encoders = perform_feature_engineering_training(df)
        
        print(f"Training data prepared. Shape: {engineered_data.shape}")
        return engineered_data, encoders
        
    except Exception as e:
        print(f"Error preparing training data: {e}")
        raise

@task(log_prints=True)
def split_training_data(df: pd.DataFrame) -> tuple:
    """
    Split prepared training data into train and validation sets.
    """
    try:
        # Assume 'target' is your target column - adjust as needed
        if 'target' not in df.columns:
            raise ValueError("Target column not found in training data")
        
        # Separate features and target
        X = df.drop(['target', 'prediction_date'], axis=1, errors='ignore')
        y = df['target']
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        return X_train, X_val, y_train, y_val
        
    except Exception as e:
        print(f"Error splitting training data: {e}")
        raise

@task(log_prints=True)
def hyperparameter_tuning(X_train: pd.DataFrame, X_val: pd.DataFrame, 
                         y_train: pd.Series, y_val: pd.Series) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning using Hyperopt and MLflow.
    """
    try:
        # Set up MLflow
        mlflow.set_experiment("xgboost_hyperparameter_tuning")
        
        # Define search space
        space = {
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'max_depth': hp.choice('max_depth', range(3, 10)),
            'min_child_weight': hp.uniform('min_child_weight', 1, 10),
            'reg_alpha': hp.uniform('reg_alpha', 0, 1),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1)
        }
        
        def objective(params):
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(params)
                
                # Train model
                model = xgb.XGBRegressor(
                    objective='reg:linear',
                    n_estimators=100,
                    random_state=42,
                    **params
                )
                
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                
                # Calculate RMSE
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                
                # Log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mse", mean_squared_error(y_val, y_pred))
                
                # Log model
                mlflow.xgboost.log_model(model, "model")
                
                return {'loss': rmse, 'status': STATUS_OK}
        
        # Run hyperparameter optimization
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,  # Adjust based on your time constraints
            trials=trials
        )
        
        # Get the best run
        best_run = min(trials.results, key=lambda x: x['loss'])
        best_rmse = best_run['loss']
        
        print(f"Best hyperparameters: {best}")
        print(f"Best RMSE: {best_rmse}")
        
        return best, best_rmse
        
    except Exception as e:
        print(f"Error in hyperparameter tuning: {e}")
        raise

@task(log_prints=True)
def train_final_model(X_train: pd.DataFrame, X_val: pd.DataFrame,
                     y_train: pd.Series, y_val: pd.Series,
                     best_params: Dict[str, Any]) -> tuple:
    """
    Train the final model with best hyperparameters.
    """
    try:
        # Combine train and validation data for final training
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])
        
        # Train final model
        final_model = xgb.XGBRegressor(
            objective='reg:linear',
            n_estimators=200,  # Increase for final model
            random_state=42,
            **best_params
        )
        
        final_model.fit(X_combined, y_combined)
        
        # Calculate final metrics
        y_pred = final_model.predict(X_val)
        final_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        print(f"Final model RMSE: {final_rmse}")
        
        return final_model, final_rmse
        
    except Exception as e:
        print(f"Error training final model: {e}")
        raise

@task(log_prints=True)
def save_model_and_update_run_id(model: xgb.XGBRegressor, reference_data: pd.DataFrame, 
                                encoders: Dict[str, LabelEncoder]) -> str:
    """
    Save the trained model, encoders, and reference data to S3, then update the run_id.
    """
    try:
        from feature_engineering import save_encoders_to_s3
        s3_client = boto3.client('s3')
        
        # Generate new run_id
        new_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_key = f"{MODEL_PREFIX}model_{new_run_id}.pkl"
        model_bytes = pickle.dumps(model)
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=model_key,
            Body=model_bytes
        )
        
        # Save encoders
        save_encoders_to_s3(encoders, new_run_id)
        
        # Save reference data for drift detection
        reference_key = f"{REFERENCE_DATA_PREFIX}reference_{new_run_id}.pkl"
        reference_bytes = pickle.dumps(reference_data)
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=reference_key,
            Body=reference_bytes
        )
        
        # Update run_id file
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=RUN_ID_FILE,
            Body=new_run_id.encode('utf-8')
        )
        
        print(f"Model, encoders, and reference data saved. Run_id updated to: {new_run_id}")
        return new_run_id
        
    except Exception as e:
        print(f"Error saving model and updating run_id: {e}")
        raise

@flow(name="weekly-model-training-flow")
def weekly_model_training_flow():
    """
    Weekly flow for retraining the XGBoost model.
    """
    try:
        # Step 1: Load all training data
        training_data = load_training_data()
        
        # Step 2: Perform feature engineering on training data
        engineered_data, encoders = prepare_training_data(training_data)
        
        # Step 3: Prepare data for training
        X_train, X_val, y_train, y_val = split_training_data(engineered_data)
        
        # Step 4: Hyperparameter tuning
        best_params, best_rmse = hyperparameter_tuning(X_train, X_val, y_train, y_val)
        
        # Step 5: Train final model
        final_model, final_rmse = train_final_model(X_train, X_val, y_train, y_val, best_params)
        
        # Step 6: Save model, encoders, and update run_id (use validation set as reference data)
        new_run_id = save_model_and_update_run_id(final_model, X_val, encoders)
        
        print(f"Weekly model training completed successfully!")
        print(f"New run_id: {new_run_id}")
        print(f"Final RMSE: {final_rmse}")
        
    except Exception as e:
        print(f"Model training flow failed with error: {e}")
        raise

if __name__ == "__main__":
    weekly_model_training_flow()
