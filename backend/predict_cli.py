"""
predict_cli.py

Usage:
  Activate venv, then:
    python predict_cli.py AAPL

It will fetch recent data and print predicted next-day return & price.
"""
import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import tensorflow as tf

SEQ_LEN = 60
FEATURES = ['Close', 'ret_1', 'ma_5', 'ma_10', 'vol_5', 'vol_10']
MODEL_DIR = "models"

def fetch_recent(ticker, days=SEQ_LEN+5):
    df = yf.download(ticker, period="120d", interval="1d", progress=False, auto_adjust=True)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    df['ret_1'] = df['Close'].pct_change(1)
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['vol_5'] = df['Volume'].rolling(window=5).mean()
    df['vol_10'] = df['Volume'].rolling(window=10).mean()
    df = df.dropna().reset_index(drop=True)
    return df

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_cli.py TICKER")
        return
    ticker = sys.argv[1].upper()
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_saved.keras"))
    input_scaler = joblib.load(os.path.join(MODEL_DIR, "input_scaler.gz"))
    y_scaler = joblib.load(os.path.join(MODEL_DIR, "target_scaler.gz"))

    df = fetch_recent(ticker)
    if len(df) < SEQ_LEN + 1:
        print("Not enough recent data to make a prediction.")
        return

    # build last sequence ending at the last available row
    last_seq = df[FEATURES].values[-SEQ_LEN:]
    last_close = df['Close'].values[-1]  # close at sequence end
    # scale
    flat = last_seq.reshape(-1, last_seq.shape[1])
    scaled_flat = input_scaler.transform(flat)
    scaled_seq = scaled_flat.reshape(1, SEQ_LEN, last_seq.shape[1])

    pred_s = model.predict(scaled_seq)
    pred_ret = y_scaler.inverse_transform(pred_s.reshape(-1,1))[0,0]
    pred_price = last_close * (1 + pred_ret)
    print(f"TICKER: {ticker}")
    print(f"Close at sequence end: {last_close[0]:.4f}")
    print(f"Predicted next-day RETURN: {pred_ret:.6f}  ({pred_ret*100:.3f} %)")
    print(f"Predicted next-day PRICE: {float(pred_price):.4f}")

if __name__ == "__main__":
    main()
