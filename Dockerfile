FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and application code
COPY lstm_model.h5 .
COPY scaler_X.pkl .
COPY scaler_y.pkl .
COPY cleaned_data.csv .
COPY pro.py .

# Run the FastAPI application
CMD ["uvicorn", "pro:app", "--host", "0.0.0.0", "--port", "8000"]