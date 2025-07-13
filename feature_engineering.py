import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic
import boto3
import pickle
import json
import os
from typing import Dict, Optional

# Configuration
S3_BUCKET = os.getenv("S3_BUCKET")
ENCODERS_PREFIX = "encoders/"

def find_nearest_stations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the nearest train stations for each row based on latitude and longitude.
    """
    # Load train station data (replace with your actual data source)
    # This could be from a CSV file, database, or API
    train_stations = {
        'brick_church': [40.76581846318419, -74.21915255150205],
        'chatham': [40.7401922968325, -74.38473871480802],
        'convent_station': [40.778934521406896, -74.44347183325733],
        'denville': [40.88348292558847, -74.48184630211975],
        'dover': [40.887548571222204, -74.55589964058176],
        'east_orange': [40.761460414532, -74.21100276083385],
        'hackettstown': [40.85215082333791, -74.83467888781628],
        'highland_avenue': [40.766972457228775, -74.24355123014908],
        'hoboken': [40.70898046045857, -74.0246430608362], #
        'lake_hopatcong': [40.904119814030835, -74.66555031993157],
        'madison': [40.757211574757704, -74.41541459013588],
        'maplewood': [40.7311582228973, -74.27530904549292],
        'millburn': [40.72583754037974, -74.303745189671],
        'morris_plains': [40.828733316576745, -74.47839671850174],
        'morristown': [40.79756715723661, -74.47460155149217],
        'mountain_station': [40.76109170550611, -74.25347657839343],
        'mount_arlington': [40.89752890181971, -74.63289981056195],
        'mount_olive': [40.90760134999547, -74.73072365897825],
        'mount_tabor': [40.8759878570467, -74.48183704548632],
        'netcong': [40.898021200833895, -74.70758666495435],
        'newark_broad': [40.74757642090106, -74.17199820501222],
        'orange': [40.77209415825383, -74.23309422970475],
        'secaucus_junction': [40.76142100515953, -74.07575294623813],
        'short_hills': [40.725313887730955, -74.3238799338488],
        'south_orange': [40.74603125221472, -74.26046288967005],
        'summit': [40.71681594165216, -74.35768690713812]
    }
    
    def find_nearest_station(row):
        user_location = (row['latitude'], row['longitude'])
        
        min_distance = 0.75  # Start at 0.75 miles
        nearest_station = "not_close"
        
        for station_name, coords in train_stations.items():
            station_location = (coords[0], coords[1])
            distance = geodesic(user_location, station_location).miles
            
            if distance < min_distance:
                min_distance = distance
                nearest_station = station_name
        
        return pd.Series([nearest_station, min_distance])
    
    df[['nearest_station', 'distance_to_station']] = df.apply(find_nearest_station, axis=1)
    return df

def save_encoders_to_s3(encoders: Dict[str, LabelEncoder], run_id: str) -> None:
    """
    Save label encoders to S3 for consistent encoding across train/predict.
    """
    try:
        s3_client = boto3.client('s3')
        
        encoders_key = f"{ENCODERS_PREFIX}encoders_{run_id}.pkl"
        encoders_bytes = pickle.dumps(encoders)
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=encoders_key,
            Body=encoders_bytes
        )
        
        print(f"Encoders saved to S3: {encoders_key}")
        
    except Exception as e:
        print(f"Error saving encoders to S3: {e}")
        raise

def load_encoders_from_s3(run_id: str) -> Dict[str, LabelEncoder]:
    """
    Load label encoders from S3 for consistent encoding.
    """
    try:
        s3_client = boto3.client('s3')
        
        encoders_key = f"{ENCODERS_PREFIX}encoders_{run_id}.pkl"
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=encoders_key)
        encoders = pickle.loads(response['Body'].read())
        
        print(f"Encoders loaded from S3: {encoders_key}")
        return encoders
        
    except Exception as e:
        print(f"Error loading encoders from S3: {e}")
        raise

def encode_categorical_variables_training(df: pd.DataFrame) -> tuple:
    """
    Encode categorical variables for training data and return both data and encoders.
    This function fits new encoders and should only be used during training.
    """
    df_encoded = df.copy()
    
    # Identify categorical columns (exclude datetime and target columns)
    categorical_columns = ['propertyType','nearest_station']
    
    # Fit label encoders to categorical columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        # Handle missing values by converting to string
        df_encoded[col] = df_encoded[col].astype(str).fillna('missing')
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    
    print(f"Fitted encoders for columns: {list(label_encoders.keys())}")
    return df_encoded, label_encoders

def encode_categorical_variables_prediction(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    """
    Encode categorical variables for prediction data using pre-fitted encoders.
    This function prevents data leakage by using encoders fitted only on training data.
    """
    df_encoded = df.copy()
    
    for col, encoder in encoders.items():
        if col in df_encoded.columns:
            # Handle missing values consistently
            df_encoded[col] = df_encoded[col].astype(str).fillna('missing')
            
            # Handle unseen categories by mapping them to a default value
            known_classes = set(encoder.classes_)
            df_encoded[col] = df_encoded[col].apply(
                lambda x: x if x in known_classes else 'missing'
            )
            
            # Transform using the fitted encoder
            df_encoded[col] = encoder.transform(df_encoded[col])
        else:
            print(f"Warning: Column {col} not found in prediction data")
    
    print(f"Applied encoders to columns: {list(encoders.keys())}")
    return df_encoded

def perform_feature_engineering_training(df: pd.DataFrame) -> tuple:
    """
    Complete feature engineering pipeline for training data.
    Returns processed data and encoders for later use.
    """
    # Step 1: Find nearest stations
    df_with_stations = find_nearest_stations(df)
    
    # Step 2: Encode categorical variables and get encoders
    df_encoded, encoders = encode_categorical_variables_training(df_with_stations)
    
    return df_encoded, encoders

def perform_feature_engineering_prediction(df: pd.DataFrame, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    """
    Complete feature engineering pipeline for prediction data.
    Uses pre-fitted encoders to prevent data leakage.
    """
    # Step 1: Find nearest stations
    df_with_stations = find_nearest_stations(df)
    
    # Step 2: Encode categorical variables using existing encoders
    df_encoded = encode_categorical_variables_prediction(df_with_stations, encoders)
    
    return df_encoded