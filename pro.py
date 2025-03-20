from fastapi import FastAPI, HTTPException
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
    """Find the closest datetime in the data index to the forecast_date."""
    abs_diff = abs(data.index - forecast_date)
    closest_datetime = data.index[abs_diff.argmin()]
    return closest_datetime

# Function to forecast features for a given date
def forecast_for_date(model, data, scaler_X, scaler_y, seq_length, forecast_date, features_to_forecast):
    """Forecast the features based on a given date."""
    
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
        
        # Get actual values for the exact date
        exact_values = data[features_to_forecast].loc[forecast_date].values.reshape(1, -1)
        actual_values = scaler_X.inverse_transform(exact_values)[0]
        
        for feature, value in zip(features_to_forecast, actual_values):
            print(f"{feature:20}: {value:,.4f}")
        
    else:
        print(f"\nDate {forecast_date} not in the dataset. Using the closest date.")
        closest_datetime = find_closest_datetime(data, forecast_date)
        forecast_date = closest_datetime
        print(f"Closest available date: {closest_datetime}")
        
        print("\nFeature values for closest date:")
        print("-" * 50)
        closest_values = data[features_to_forecast].loc[closest_datetime].values.reshape(1, -1)
        original_values = scaler_X.inverse_transform(closest_values)[0]
        
        for feature, value in zip(features_to_forecast, original_values):
            print(f"{feature:20}: {value:,.4f}")
    
    print("-" * 50)

    # Get historical sequence for prediction
    closest_datetime_index = data.index.get_loc(forecast_date)
    historical_sequence = data[features_to_forecast].iloc[closest_datetime_index-seq_length:closest_datetime_index].values
    historical_sequence = historical_sequence.reshape(1, seq_length, len(features_to_forecast))

    # Make prediction
    predicted = model.predict(historical_sequence, verbose=0)
    predicted_original_scale = scaler_y.inverse_transform(predicted)

    return predicted_original_scale[0, 0]

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
        predicted_dc_power = forecast_for_date(
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