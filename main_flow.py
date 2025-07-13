import os
import pandas as pd
import boto3
import requests
from datetime import datetime, timedelta
from prefect import flow, task
from prefect.blocks.system import Secret
from prefect_aws import AwsBatch
import xgboost as xgb
from evidently.metric_preset import DataDriftPreset, DataSummaryPreset
from evidently.report import Report
import pickle
import json
from typing import Dict, Any

from feature_engineering import perform_feature_engineering_training
from utils import send_email_with_attachment

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET")
TRAINING_DATA_BUCKET = os.getenv("TRAINING_DATA_BUCKET")
RUN_ID_FILE = "run_id.txt"
MODEL_PREFIX = "models/"
REFERENCE_DATA_PREFIX = "reference_data/"
TRAINING_DATA_PREFIX = "training_data/"

@task(log_prints=True)
def fetch_api_data(api_key: str) -> pd.DataFrame:
    """
    Fetch data from API using the provided API key.
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Replace with your actual API endpoint
        response = requests.get(
            "https://your-api-endpoint.com/data",
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data)
        
        print(f"Successfully fetched {len(df)} rows from API")
        return df
        
    except Exception as e:
        print(f"Error fetching API data: {e}")
        raise

@task(log_prints=True)
def get_run_id() -> str:
    """
    Get the current run_id from S3.
    """
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=RUN_ID_FILE)
        run_id = response['Body'].read().decode('utf-8').strip()
        print(f"Retrieved run_id: {run_id}")
        return run_id
        
    except Exception as e:
        print(f"Error retrieving run_id: {e}")
        raise

@task(log_prints=True)
def perform_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering on the raw data for prediction.
    """
    try:
        print("Starting feature engineering for prediction...")
        
        # Get current run_id to load the correct encoders
        run_id = get_run_id()
        
        # Load encoders from S3
        from feature_engineering import load_encoders_from_s3, perform_feature_engineering_prediction
        encoders = load_encoders_from_s3(run_id)
        
        # Perform feature engineering with pre-fitted encoders
        df_engineered = perform_feature_engineering_prediction(df, encoders)
        
        print(f"Feature engineering completed. Shape: {df_engineered.shape}")
        return df_engineered
        
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        raise

@task(log_prints=True)
def load_model_and_reference_data(run_id: str) -> tuple:
    """
    Load the XGBoost model and reference data from S3.
    """
    try:
        s3_client = boto3.client('s3')
        
        # Load model
        model_key = f"{MODEL_PREFIX}model_{run_id}.pkl"
        model_response = s3_client.get_object(Bucket=S3_BUCKET, Key=model_key)
        model = pickle.loads(model_response['Body'].read())
        
        # Load reference data for drift detection
        reference_key = f"{REFERENCE_DATA_PREFIX}reference_{run_id}.pkl"
        reference_response = s3_client.get_object(Bucket=S3_BUCKET, Key=reference_key)
        reference_data = pickle.loads(reference_response['Body'].read())
        
        print(f"Successfully loaded model and reference data for run_id: {run_id}")
        return model, reference_data
        
    except Exception as e:
        print(f"Error loading model and reference data: {e}")
        raise

@task(log_prints=True)
def generate_predictions(model: xgb.XGBRegressor, df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions using the XGBoost model.
    """
    try:
        # Assume df already has the correct features for prediction
        predictions = model.predict(df)
        
        # Add predictions to dataframe
        df_with_predictions = df.copy()
        df_with_predictions['prediction'] = predictions
        df_with_predictions['prediction_date'] = datetime.now()
        
        print(f"Generated {len(predictions)} predictions")
        return df_with_predictions
        
    except Exception as e:
        print(f"Error generating predictions: {e}")
        raise

@task(log_prints=True)
def generate_evidently_report(current_data: pd.DataFrame, reference_data: pd.DataFrame) -> str:
    """
    Generate Evidently model and data monitoring report.
    """
    try:
        # Create report with specified presets
        report = Report(metrics=[
            DataDriftPreset(),
            DataSummaryPreset()
        ])
        
        # Run the report
        report.run(reference_data=reference_data, current_data=current_data)
        
        # Save report as HTML
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"evidently_report_{timestamp}.html"
        report.save_html(report_filename)
        
        print(f"Evidently report generated: {report_filename}")
        return report_filename
        
    except Exception as e:
        print(f"Error generating Evidently report: {e}")
        raise

@task(log_prints=True)
def append_to_training_data(predictions_df: pd.DataFrame):
    """
    Append daily predictions to the training dataset in S3.
    """
    try:
        s3_client = boto3.client('s3')
        
        # Save daily data to training bucket
        timestamp = datetime.now().strftime("%Y%m%d")
        s3_key = f"{TRAINING_DATA_PREFIX}daily_data_{timestamp}.csv"
        
        # Convert to CSV and upload
        csv_buffer = predictions_df.to_csv(index=False)
        s3_client.put_object(
            Bucket=TRAINING_DATA_BUCKET,
            Key=s3_key,
            Body=csv_buffer,
            ContentType='text/csv'
        )
        
    except Exception as e:
        print(f"Error appending to training data: {e}")
        raise

@task(log_prints=True)
def save_predictions_to_s3(predictions_df: pd.DataFrame, run_id: str) -> str:
    """
    Save predictions to S3 and return the S3 key.
    """
    try:
        s3_client = boto3.client('s3')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"predictions/predictions_{run_id}_{timestamp}.csv"
        
        # Convert to CSV and upload
        csv_buffer = predictions_df.to_csv(index=False)
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer,
            ContentType='text/csv'
        )
        
        print(f"Predictions saved to S3: {s3_key}")
        return s3_key
        
    except Exception as e:
        print(f"Error saving predictions to S3: {e}")
        raise

@task(log_prints=True)
def email_predictions(predictions_df: pd.DataFrame, report_filename: str):
    """
    Email predictions and monitoring report using AWS SES.
    """
    try:
        # Save predictions to temporary CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_filename = f"predictions_{timestamp}.csv"
        predictions_df.to_csv(predictions_filename, index=False)
        
        # Email configuration
        subject = f"Daily Predictions Report - {datetime.now().strftime('%Y-%m-%d')}"
        body = f"""
        Daily prediction run completed successfully.
        
        Run details:
        - Timestamp: {datetime.now()}
        - Number of predictions: {len(predictions_df)}
        - Attachments: predictions CSV and monitoring report
        
        Please find the predictions and monitoring report attached.
        """
        TO_ADDRESS = os.getenv('TO_ADDRESS','')
        recipients = [TO_ADDRESS]  # Replace with actual recipients
        
        attachments = [predictions_filename, report_filename]
        
        send_email_with_attachment(
            subject=subject,
            body=body,
            recipients=recipients,
            attachments=attachments
        )
        
        # Clean up temporary files
        os.remove(predictions_filename)
        os.remove(report_filename)
        
        print("Email sent successfully")
        
    except Exception as e:
        print(f"Error sending email: {e}")
        raise

@flow(name="daily-prediction-flow")
def daily_prediction_flow():
    """
    Main Prefect flow for daily predictions.
    """
    try:
        # Get API key from Prefect Secret block
        api_key = Secret.load("api-key").get()
        
        # Step 1: Fetch data from API
        raw_data = fetch_api_data(api_key)
        
        # Step 2: Get current run_id
        run_id = get_run_id()
        
        # Step 3: Perform feature engineering
        engineered_data = perform_feature_engineering(raw_data)
        
        # Step 4: Load model and reference data
        model, reference_data = load_model_and_reference_data(run_id)
        
        # Step 5: Generate predictions
        predictions = generate_predictions(model, engineered_data)
        
        # Step 6: Generate Evidently report
        report_filename = generate_evidently_report(engineered_data, reference_data)
        
        # Step 7: Save predictions to S3
        s3_key = save_predictions_to_s3(predictions, run_id)
        
        # Step 8: Append to training data
        append_to_training_data(predictions)
        
        # Step 9: Email predictions and report
        email_predictions(predictions, report_filename)
        
        print(f"Daily prediction flow completed successfully!")
        
    except Exception as e:
        print(f"Flow failed with error: {e}")
        raise

if __name__ == "__main__":
    daily_prediction_flow()