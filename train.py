import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
def prepare_data():
    file_path = 'cleaned_data.csv'
    data = pd.read_csv(file_path)
    
    # Convert datetime with explicit format and dayfirst=True
    try:
        data['datetime'] = pd.to_datetime(
            data['Date'] + ' ' + data['Time'],
            format='%d-%m-%Y %H:%M:%S',  # Specify DD-MM-YYYY format
            dayfirst=True  # Tell pandas that day comes before month
        )
    except ValueError as e:
        print(f"Error during datetime conversion: {e}")
        # Fallback to more flexible parsing if needed
        data['datetime'] = pd.to_datetime(
            data['Date'] + ' ' + data['Time'],
            dayfirst=True
        )
    
    data.set_index('datetime', inplace=True)
    data.drop(['Date', 'Time'], axis=1, inplace=True)
    
    return data

# Create sequences for LSTM
def create_sequences(data, seq_length, features, target):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i+seq_length)][features].values
        y = data.iloc[i+seq_length][target]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_and_save_model():
    # Define features and target
    features_to_forecast = ['AC frequency', 'AC voltage', 'DCLink Voltage', 'Energy today',
                          'Output current', 'Total Energy', 'output power', 'DC Current',
                          'Pyranometer', 'Temperature', 'Power Factor']
    target_variable = 'DC Power'
    
    # Prepare data
    data = prepare_data()
    
    # Scale data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    data[features_to_forecast] = scaler_X.fit_transform(data[features_to_forecast])
    data[[target_variable]] = scaler_y.fit_transform(data[[target_variable]])
    
    # Create sequences
    seq_length = 10
    X, y = create_sequences(data, seq_length, features_to_forecast, target_variable)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Define and train model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    
    # Use tf.keras.losses.MeanSquaredError() instead of 'mse'
    model.compile(
        optimizer=Adam(learning_rate=0.0005), 
        loss=tf.keras.losses.MeanSquaredError()
    )
    
    # Add custom_objects when saving
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
    
    # Save model with custom objects
    model.save('lstm_model.h5', save_format='h5')
    
    # Evaluate model
    predicted = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicted))
    mae = mean_absolute_error(y_test, predicted)
    r2 = r2_score(y_test, predicted)
    
    print(f'LSTM RMSE: {rmse:.3f}')
    print(f'LSTM MAE: {mae:.3f}')
    print(f'LSTM R2 Score: {r2:.3f}')
    
    # Save model and scalers
    save_model(model, 'lstm_model.h5')
    with open('scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)

if __name__ == "__main__":
    train_and_save_model()