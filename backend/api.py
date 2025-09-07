from fastapi import FastAPI, Query
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

app = FastAPI(title="Stock Price Predictor")

# Utility: build & train model quickly
def train_and_predict(ticker: str):
    df = yf.download(ticker, start="2015-01-01", end="2023-01-01")
    if df.empty:
        return {"error": f"No data found for {ticker}"}

    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    seq_len = 60
    x_train, y_train = [], []
    for i in range(seq_len, len(scaled_data)):
        x_train.append(scaled_data[i-seq_len:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(x_train.shape[1],1)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=3, batch_size=32, verbose=0)

    # Test predictions
    test_data = scaled_data[-(seq_len+50):]
    x_test, y_test = [], data[-50:]
    for i in range(seq_len, len(test_data)):
        x_test.append(test_data[i-seq_len:i, 0])
    x_test = np.array(x_test).reshape(-1, seq_len, 1)

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return {
        "ticker": ticker,
        "last_actual_price": round(float(y_test[-1]), 2),
        "next_predicted_price": round(float(predictions[-1]), 2)
    }

@app.get("/predict")
def predict(ticker: str = Query(..., description="Stock symbol, e.g., AAPL or INFY.NS")):
    return train_and_predict(ticker)
