from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import pickle
import os

# Configure TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load pre-trained model and scalers
try:
    # Define custom objects for loading the model
    custom_objects = {
        'MeanSquaredError': tf.keras.losses.MeanSquaredError
    }
    
    model = load_model('lstm_model.h5', custom_objects=custom_objects)
    with open('scaler_X.pkl', 'rb') as f:
        scaler_X = pickle.load(f)
    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
    
    # Load dataset for reference
    data = pd.read_csv('cleaned_data.csv')
    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d-%m-%Y %H:%M:%S')
    data.set_index('datetime', inplace=True)
    data.drop(['Date', 'Time'], axis=1, inplace=True)
except Exception as e:
    print(f"Error loading model and data: {e}")
    raise

# Constants
features_to_forecast = ['AC frequency', 'AC voltage', 'DCLink Voltage', 'Energy today',
                       'Output current', 'Total Energy', 'output power', 'DC Current',
                       'Pyranometer', 'Temperature', 'Power Factor']
seq_length = 10

# Function to find closest datetime in the data index
def find_closest_datetime(data, forecast_date):
    """Find the closest datetime by matching date and time separately."""
    
    if not isinstance(forecast_date, pd.Timestamp):
        forecast_date = pd.to_datetime(forecast_date)
    
    # Get all timestamps and target components
    timestamps = data.index
    target_hour = forecast_date.hour
    
    # Define time periods (in hours)
    morning = range(5, 12)      # 5:00 - 11:59
    afternoon = range(12, 17)   # 12:00 - 16:59
    evening = range(17, 21)     # 17:00 - 20:59
    night = list(range(21, 24)) + list(range(0, 5))  # 21:00 - 4:59
    
    # Determine target period
    if target_hour in morning:
        target_period = morning
    elif target_hour in afternoon:
        target_period = afternoon
    elif target_hour in evening:
        target_period = evening
    else:
        target_period = night
    
    # Convert timestamps to pandas DatetimeIndex if not already
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.DatetimeIndex(timestamps)
    
    # Find dates within ±5 days
    date_mask = (timestamps >= forecast_date - pd.Timedelta(days=5)) & \
                (timestamps <= forecast_date + pd.Timedelta(days=5))
    nearby_dates = timestamps[date_mask]
    
    if len(nearby_dates) > 0:
        # Filter for same time period
        same_period_mask = pd.Series([ts.hour in target_period for ts in nearby_dates])
        period_matches = nearby_dates[same_period_mask]
        
        if len(period_matches) > 0:
            # Find closest time within period matches
            time_diffs = abs(period_matches - forecast_date)
            closest_datetime = period_matches[time_diffs.argmin()]
        else:
            # If no period matches, use closest nearby date
            time_diffs = abs(nearby_dates - forecast_date)
            closest_datetime = nearby_dates[time_diffs.argmin()]
            
        # Calculate differences for display
        time_diff_hours = abs(closest_datetime - forecast_date).total_seconds() / 3600
        date_diff_days = abs(closest_datetime.date() - forecast_date.date()).days
        
        print(f"\nRequested datetime: {forecast_date}")
        print(f"Closest match found:")
        print(f"Date: {closest_datetime.date()} (diff: {date_diff_days} days)")
        print(f"Time: {closest_datetime.time()} (diff: {time_diff_hours:.2f} hours)")
    else:
        # If no nearby dates, find closest match in same time period
        same_period_mask = pd.Series([ts.hour in target_period for ts in timestamps])
        period_matches = timestamps[same_period_mask]
        
        if len(period_matches) > 0:
            time_diffs = abs(period_matches - forecast_date)
            closest_datetime = period_matches[time_diffs.argmin()]
        else:
            time_diffs = abs(timestamps - forecast_date)
            closest_datetime = timestamps[time_diffs.argmin()]
        
        print(f"\nNo nearby dates found within ±5 days.")
        print(f"Using closest available datetime: {closest_datetime}")
    
    return closest_datetime

# Function to forecast features for a given date
def forecast_for_date(model, data, scaler_X, scaler_y, seq_length, forecast_date, features_to_forecast):
    """Forecast the features based on a given date."""
    
    # Initialize actual_values variable
    actual_values = None
    
    # Print dataset date range
    print(f"\nDataset covers: {data.index.min()} to {data.index.max()}")

    # Ensure forecast_date is a datetime object
    if not isinstance(forecast_date, pd.Timestamp):
        forecast_date = pd.to_datetime(forecast_date)

    # Check if the forecast_date is in the data index
    if forecast_date in data.index:
        print(f"\nExact match found for date: {forecast_date}")
        print("\nFeature values for requested date:")
        print("-" * 50)
        
        # Get raw values without scaling
        exact_values = data[features_to_forecast].loc[forecast_date].values.reshape(1, -1)
        # Don't apply inverse_transform here since these are raw values
        actual_values = exact_values[0]
        
        for feature, value in zip(features_to_forecast, actual_values):
            print(f"{feature:20}: {value:,.4f}")
        
    else:
        print(f"\nDate {forecast_date} not in the dataset. Using the closest date.")
        closest_datetime = find_closest_datetime(data, forecast_date)
        forecast_date = closest_datetime
        print(f"Closest available date: {closest_datetime}")
        
        print("\nFeature values for closest date:")
        print("-" * 50)
        # Get raw values without scaling
        closest_values = data[features_to_forecast].loc[closest_datetime].values.reshape(1, -1)
        # Don't apply inverse_transform here since these are raw values
        actual_values = closest_values[0]
        
        for feature, value in zip(features_to_forecast, actual_values):
            print(f"{feature:20}: {value:,.4f}")
    
    print("-" * 50)

    # Get historical sequence for prediction
    closest_datetime_index = data.index.get_loc(forecast_date)
    historical_sequence = data[features_to_forecast].iloc[closest_datetime_index-seq_length:closest_datetime_index].values
    historical_sequence = historical_sequence.reshape(1, seq_length, len(features_to_forecast))

    # Make prediction
    predicted = model.predict(historical_sequence, verbose=0)
    predicted_original_scale = scaler_y.inverse_transform(predicted)

    # Create features dictionary with actual values
    feature_values = {
        feature: float(value) 
        for feature, value in zip(features_to_forecast, actual_values)
    }

    return predicted_original_scale[0, 0], feature_values

# Define request body schema using Pydantic
class ForecastRequest(BaseModel):
    forecast_date: str  # Date in 'YYYY-MM-DD HH:MM:SS' format

# Define response schema
class ForecastResponse(BaseModel):
    forecast_date: str
    predicted_dc_power: float
    features: dict

# Define the API endpoint
@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    try:
        forecast_date = pd.to_datetime(request.forecast_date)
        
        # Get the feature values before making prediction
        if forecast_date in data.index:
            values = data[features_to_forecast].loc[forecast_date].values.reshape(1, -1)
        else:
            closest_datetime = find_closest_datetime(data, forecast_date)
            values = data[features_to_forecast].loc[closest_datetime].values.reshape(1, -1)
        
        # Get original scale values
        original_values = scaler_X.inverse_transform(values)[0]
        
        # Create features dictionary
        feature_values = {
            feature: float(value) 
            for feature, value in zip(features_to_forecast, original_values)
        }

        # Get prediction
        predicted_dc_power, feature_values = forecast_for_date(
            model=model,
            data=data,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            seq_length=seq_length,
            forecast_date=forecast_date,
            features_to_forecast=features_to_forecast
        )

        # Return the enhanced response
        return ForecastResponse(
            forecast_date=request.forecast_date,
            predicted_dc_power=predicted_dc_power,
            features=feature_values
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during forecasting: {str(e)}")

# Run the server
if __name__ == "__main__":
    import uvicorn
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    # Configure host and port
    port = int(os.getenv("PORT", 8000))
    
    # Run with modified settings
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,  # Reduce workers to avoid GPU conflicts
        log_level="info",
        timeout_keep_alive=30,
        access_log=True
    )
