from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
import os
import tensorflow as tf
warnings.filterwarnings('ignore')

# Configure TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Initialize FastAPI app
app = FastAPI()

# Load the dataset
file_path = 'cleaned_data.csv'
data = pd.read_csv(file_path)

# Convert 'Date' and 'Time' columns into a single 'datetime' column
try:
    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d-%m-%Y %H:%M:%S')
except ValueError as e:
    print(f"Error during datetime conversion: {e}. Trying alternative format.")
    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])  # Let pandas infer the format
data.set_index('datetime', inplace=True)
data.drop(['Date', 'Time'], axis=1, inplace=True)

# Define features and target variable
features_to_forecast = ['AC frequency', 'AC voltage', 'DCLink Voltage', 'Energy today',
                        'Output current', 'Total Energy', 'output power', 'DC Current',
                        'Pyranometer', 'Temperature', 'Power Factor']
target_variable = 'DC Power'

# Scale the data
scaler_X = MinMaxScaler()
data[features_to_forecast] = scaler_X.fit_transform(data[features_to_forecast])

scaler_y = MinMaxScaler()
data[[target_variable]] = scaler_y.fit_transform(data[[target_variable]])

# Function to create sequences for LSTM
def create_sequences(data, seq_length, features, target):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i+seq_length)][features].values
        y = data.iloc[i+seq_length][target]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Set sequence length
seq_length = 10

# Create sequences
X, y = create_sequences(data, seq_length, features_to_forecast, target_variable)

# Prepare training data for LSTM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(1))  # Output layer for DC Power prediction

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mse')

# Train the LSTM model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the LSTM model
predicted = model.predict(X_test)

# Evaluate metrics
rmse = np.sqrt(mean_squared_error(y_test, predicted))
mae = mean_absolute_error(y_test, predicted)
r2 = r2_score(y_test, predicted)

print(f'LSTM RMSE: {rmse:.3f}')
print(f'LSTM MAE: {mae:.3f}')
print(f'LSTM R2 Score: {r2:.3f}')

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
