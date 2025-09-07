# 1. Install dependencies (run these once in your terminal)
# pip install yfinance pandas numpy scikit-learn tensorflow

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# -----------------------------
# STEP 1: User Input
# -----------------------------
ticker = input("Enter stock ticker (e.g., AAPL for Apple or INFY.NS for Infosys India): ")

# Download stock data
df = yf.download(ticker, start="2015-01-01", end="2023-01-01")

if df.empty:
    raise ValueError(f"⚠️ No data found for ticker '{ticker}'. Please check Yahoo Finance for the correct symbol.")

print(f"✅ Successfully downloaded {len(df)} rows for {ticker}")

# -----------------------------
# STEP 2: Preprocessing
# -----------------------------
data = df[['Close']].values  # only closing price
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

sequence_length = 60
x_train, y_train = [], []

for i in range(sequence_length, len(scaled_data)):
    x_train.append(scaled_data[i-sequence_length:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# -----------------------------
# STEP 3: Model
# -----------------------------
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=32, return_sequences=False))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# -----------------------------
# STEP 4: Training
# -----------------------------
print("⏳ Training model...")
model.fit(x_train, y_train, epochs=5, batch_size=32)  # keep epochs small for quick test

# -----------------------------
# STEP 5: Testing / Predictions
# -----------------------------
# Use last 20% of data for testing
train_size = int(len(scaled_data)*0.8)
test_data = scaled_data[train_size - sequence_length:]
x_test, y_test = [], data[train_size:]

for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# -----------------------------
# STEP 6: Plot Results
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(data, color="blue", label=f"Actual {ticker} Price")
plt.plot(range(train_size, train_size+len(predictions)), predictions, color="red", label="Predicted Price")
plt.title(f"{ticker} Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()
