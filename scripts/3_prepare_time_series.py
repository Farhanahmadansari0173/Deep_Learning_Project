"""
Script 3: Time-Series Data Preparation

Neural Networks like LSTMs don't just take single rows of data. They take "sequences" or "windows".
For example, to predict Day 10's stock price, the LSTM needs to look at the sequence of Days 1 to 9.
This script merges our Financial NLP Sentiment with the numerical stock data, scales the values, 
and generates these sliding windows for PyTorch.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
SEQ_LENGTH = 10 # The LSTM will look back 10 days to predict the next day

def create_sequences(data, seq_length):
    xs = []
    ys = []
    # Loop through the data to create overlapping windows
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0] # Index 0 is the 'Close' price we want to predict
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("Loading datasets...")
    stock_df = pd.read_csv(os.path.join(DATA_DIR, "stock_prices.csv"))
    sentiment_df = pd.read_csv(os.path.join(PROCESSED_DIR, "daily_sentiment.csv"))
    
    # Merge on the Date column
    df = pd.merge(stock_df, sentiment_df, on='Date', how='left')
    
    # If there are weekends or holidays with no news, fill the sentiment with 0.0 (Neutral)
    df['Sentiment'] = df['Sentiment'].fillna(0.0)
    
    # Sort by date chronologically
    df = df.sort_values('Date')
    
    # We will use Close, Volume, and our new NLP Sentiment feature
    # The target variable we want to predict MUST be the first column for our indexing to work
    features = ['Close', 'Volume', 'Sentiment']
    data_to_scale = df[features].values
    
    print("Scaling data to range [0, 1] for neural network stability...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_to_scale)
    
    # Save the scaler so we can un-scale the predictions later to actual dollar amounts
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"Creating time-series sequences (window size = {SEQ_LENGTH} days)...")
    X, y = create_sequences(scaled_data, SEQ_LENGTH)
    
    # Train/Test Split (Time Series must be split chronologically, not randomly!)
    train_size = int(len(X) * 0.8)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Save the prepared sequences for PyTorch
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)
    
    print(f"Prepared {len(X_train)} training sequences and {len(X_test)} test sequences.")
    print(f"Data saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
