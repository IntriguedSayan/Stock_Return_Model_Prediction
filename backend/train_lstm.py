"""
train_lstm.py

Train a time-series LSTM to predict next-day RETURN for a stock ticker.
Saves:
  - TensorFlow model -> models/lstm_saved (SavedModel)
  - input scaler -> models/input_scaler.gz
  - target scaler -> models/target_scaler.gz
"""

import os
import random
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Config
# -------------------------
TICKER = "AAPL" 
PERIOD = "5y"             # data history to download (yfinance)
INTERVAL = "1d"
SEQ_LEN = 60              # number of past trading days used as input
FEATURES = ['Close', 'ret_1', 'ma_5', 'ma_10', 'vol_5', 'vol_10']  # features used
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Utilities / Data Prep
# -------------------------
def fetch_data(ticker=TICKER, period=PERIOD, interval=INTERVAL):
    # auto_adjust=True corrects for splits/dividends
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    return df

def add_features(df):
    df = df.copy()
    # past daily return
    df['ret_1'] = df['Close'].pct_change(1)
    # simple moving averages
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    # volume averages
    df['vol_5'] = df['Volume'].rolling(window=5).mean()
    df['vol_10'] = df['Volume'].rolling(window=10).mean()
    # drop rows with NaN (initial rows due to rolling windows)
    df = df.dropna().reset_index(drop=True)
    return df

def create_sequences(df, seq_len=SEQ_LEN):
    """
    For each time step i where seq ends at i (inclusive),
    the target is next-day return: (Close[i+1] - Close[i]) / Close[i]
    So we can only form samples up to len(df) - 2 as i+1 must exist.
    Returns:
      X: (n_samples, seq_len, n_features)
      y: (n_samples, 1)  -> next-day return
      close_at_seq_end: (n_samples,) -> Close at the final day of the sequence (used to compute price predictions)
      idx_dates: dates corresponding to the sequence end (for reference)
    """
    values = df[FEATURES].values
    closes = df['Close'].values
    dates = df.index if hasattr(df, "index") else np.arange(len(df))
    X, y, close_at_end, idx_dates = [], [], [], []
    # we need i from seq_len-1 to len(df)-2 (so i+1 exists)
    for end_idx in range(seq_len - 1, len(df) - 1):
        start_idx = end_idx - (seq_len - 1)
        seq = values[start_idx:end_idx+1]   # inclusive end_idx -> length seq_len
        # target uses next day's close:
        next_ret = (closes[end_idx + 1] - closes[end_idx]) / closes[end_idx]
        X.append(seq)
        y.append(next_ret)
        close_at_end.append(closes[end_idx])
        idx_dates.append(dates[end_idx])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    close_at_end = np.array(close_at_end).reshape(-1, 1)
    return X, y, close_at_end, idx_dates

# -------------------------
# Model builder
# -------------------------
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))  # predict next-day return (can be negative/positive)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# -------------------------
# Main training flow
# -------------------------
def main():
    print("Fetching data for", TICKER)
    df = fetch_data()
    df = add_features(df)
    print("Data after features:", df.shape)

    # create sequences and targets
    X, y, closes, idx_dates = create_sequences(df, seq_len=SEQ_LEN)
    print("Sequences shape:", X.shape, "Targets shape:", y.shape)

    # time-based split: train 80%, val 10%, test 10%
    n = len(X)
    test_size = int(0.1 * n)
    val_size = int(0.1 * n)
    train_end = n - test_size - val_size

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:train_end+val_size], y[train_end:train_end+val_size]
    X_test, y_test = X[train_end+val_size:], y[train_end+val_size:]
    closes_test = closes[train_end+val_size:]
    dates_test = idx_dates[train_end+val_size:]

    print("Train/Val/Test sizes:", len(X_train), len(X_val), len(X_test))

    # Flatten sequences for scaler fitting: (n_samples*seq_len, n_features)
    n_samples, seq_len, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)

    # Fit scaler on training data only
    input_scaler = MinMaxScaler()
    input_scaler.fit(X_train_flat)

    # Transform all sets
    def scale_X(X_arr):
        ns, sl, nf = X_arr.shape
        flat = X_arr.reshape(-1, nf)
        scaled_flat = input_scaler.transform(flat)
        return scaled_flat.reshape(ns, sl, nf)

    X_train_s = scale_X(X_train)
    X_val_s = scale_X(X_val)
    X_test_s = scale_X(X_test)

    # Scale target (returns): standard scaler is fine (mean~0)
    y_scaler = StandardScaler()
    y_scaler.fit(y_train)
    y_train_s = y_scaler.transform(y_train)
    y_val_s = y_scaler.transform(y_val)
    y_test_s = y_scaler.transform(y_test)

    # build model
    model = build_lstm(input_shape=(seq_len, n_features))
    model.summary()

    # training
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=100,
        batch_size=32,
        callbacks=[es],
        verbose=2
    )

    # Evaluate on test set (transform predictions back to return scale)
    y_pred_s = model.predict(X_test_s)
    y_pred = y_scaler.inverse_transform(y_pred_s)
    y_true = y_test

    # metrics on returns
    rmse_ret = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_ret = mean_absolute_error(y_true, y_pred)
    print(f"Test Returns RMSE: {rmse_ret:.6f}, MAE: {mae_ret:.6f}")

    # convert returns to prices for price-level metrics
    # predicted_price = close_at_seq_end * (1 + predicted_return)
    pred_prices = closes_test * (1 + y_pred)
    true_prices = closes_test * (1 + y_true)
    rmse_price = np.sqrt(mean_squared_error(true_prices, pred_prices))
    mae_price = mean_absolute_error(true_prices, pred_prices)
    print(f"Test Price RMSE: {rmse_price:.4f}, MAE: {mae_price:.4f}")

    # Save model & scalers
    print("Saving model and scalers to", MODEL_DIR)
    model.save(os.path.join(MODEL_DIR, "lstm_saved.keras"))  # SavedModel dir
    joblib.dump(input_scaler, os.path.join(MODEL_DIR, "input_scaler.gz"))
    joblib.dump(y_scaler, os.path.join(MODEL_DIR, "target_scaler.gz"))

    # Optionally: save a small CSV with test predictions for quick visualization
    out_df = pd.DataFrame({
        'date': dates_test,
        'close_at_seq_end': closes_test.flatten(),
        'true_next_return': y_true.flatten(),
        'pred_next_return': y_pred.flatten(),
        'true_next_price': true_prices.flatten(),
        'pred_next_price': pred_prices.flatten()
    })
    out_df.to_csv(os.path.join(MODEL_DIR, "test_predictions.csv"), index=False)
    print("Saved test predictions to models/test_predictions.csv")
    print("Done.")

if __name__ == "__main__":
    main()
