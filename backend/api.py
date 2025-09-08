from fastapi import FastAPI, Query, HTTPException
import httpx
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from datetime import datetime, timedelta
import joblib
from functools import lru_cache

app = FastAPI(title="Stock Price Predictor")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Constants
SEQ_LEN = 60  # Keep original sequence length for better pattern recognition
CACHE_DURATION = timedelta(hours=24)  # How long to keep models in memory
MODEL_CACHE = {}
TRAINING_YEARS = 3  # Use 3 years of data for a balance of history and relevance

def get_model_path(ticker: str):
    return f"models/{ticker.replace('.', '_')}"

@lru_cache(maxsize=100)
def get_cached_stock_data(ticker: str):
    """Cache stock data to avoid frequent downloads"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=TRAINING_YEARS*365)  # Use 3 years of data
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data for {ticker}: {str(e)}")

def train_model(ticker: str, data: np.ndarray):
    """Train a new model for the given ticker"""
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    
    # Prepare training data
    x_train, y_train = [], []
    for i in range(SEQ_LEN, len(scaled_data)):
        x_train.append(scaled_data[i-SEQ_LEN:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # More sophisticated model architecture
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN,1)),
        Dropout(0.2),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer="adam", loss="mse")
    
    # Train with early stopping logic
    best_loss = float('inf')
    patience = 0
    max_patience = 2
    min_epochs = 3
    max_epochs = 5
    
    for epoch in range(max_epochs):
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        current_loss = history.history['loss'][0]
        
        if epoch + 1 >= min_epochs:
            if current_loss < best_loss:
                best_loss = current_loss
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    break
    
    # Save model and scaler
    model_path = get_model_path(ticker)
    model.save(f"{model_path}_model.keras")
    joblib.dump(scaler, f"{model_path}_scaler.joblib")
    
    return model, scaler

def load_or_train_model(ticker: str, data: np.ndarray):
    """Load existing model or train new one if needed"""
    model_path = get_model_path(ticker)
    
    # Check if model is in memory cache
    if ticker in MODEL_CACHE:
        model, scaler, timestamp = MODEL_CACHE[ticker]
        if datetime.now() - timestamp < CACHE_DURATION:
            return model, scaler

    # Try to load saved model
    try:
        if os.path.exists(f"{model_path}_model.keras"):
            model = load_model(f"{model_path}_model.keras")
            scaler = joblib.load(f"{model_path}_scaler.joblib")
        else:
            model, scaler = train_model(ticker, data)
        
        # Update cache
        MODEL_CACHE[ticker] = (model, scaler, datetime.now())
        return model, scaler
    except:
        # If loading fails, train new model
        return train_model(ticker, data)

def predict_price(ticker: str, model, scaler, recent_data: np.ndarray):
    """Make prediction using loaded model"""
    scaled_data = scaler.transform(recent_data)
    
    # Prepare input sequence
    x_test = []
    x_test.append(scaled_data[-SEQ_LEN:, 0])
    x_test = np.array(x_test).reshape(-1, SEQ_LEN, 1)
    
    # Make prediction
    pred = model.predict(x_test, verbose=0)
    pred = scaler.inverse_transform(pred)
    
    return float(pred[0, 0])

def train_and_predict(ticker: str):
    """Main function to handle model training and prediction"""
    try:
        # Get historical data
        df = get_cached_stock_data(ticker)
        data = df[['Close']].values
        
        # Load or train model
        model, scaler = load_or_train_model(ticker, data)
        
        # Get last actual price and prediction
        last_price = float(data[-1][0])
        next_price = predict_price(ticker, model, scaler, data)
        
        return {
            "ticker": ticker,
            "last_actual_price": round(last_price, 2),
            "next_predicted_price": round(next_price, 2)
        }
    except HTTPException as he:
        return {"error": he.detail}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.get("/predict")
def predict(ticker: str = Query(..., description="Stock symbol, e.g., AAPL or INFY.NS")):
    return train_and_predict(ticker)

@app.get("/search")
async def search_stocks(query: str):
    """Search stocks by company name using Yahoo Finance unofficial API."""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=5&newsCount=0"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        if response.status_code == 200:
            res = response.json()
        else:
            return {"error": "Failed to fetch data from Yahoo Finance"}

    results = []
    for quote in res.get("quotes", []):
        results.append({
            "symbol": quote.get("symbol"),
            "name": quote.get("shortname", quote.get("longname", "")),
            "exchange": quote.get("exchDisp", "")
        })

    return results

