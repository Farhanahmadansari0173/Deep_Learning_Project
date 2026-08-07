# 📈 Financial Sentiment & Stock Prediction (Deep Learning)

My name is **Farhan Ahmad Ansari**, and I built this repository to tackle one of the most exciting challenges in AI: **Multimodal Deep Learning for Financial Forecasting**.

This project combines **Natural Language Processing (NLP)** and **Time-Series Forecasting** to predict stock market movements. Instead of just looking at historical numbers, this AI actually reads financial news, understands the sentiment, and uses that context to make better predictions.

## 🧠 The Architecture

This project is built using a dual-pipeline approach:

1. **The NLP Pipeline (Transformers)**:
   - Traditional NLP models don't understand financial terminology (e.g., "short" or "bear"). 
   - I used **FinBERT** (a HuggingFace Transformer pre-trained on financial text) to analyze thousands of financial news headlines.
   - The model extracts a daily "Sentiment Score" (Positive/Negative/Neutral) based on market news.

2. **The Time-Series Pipeline (LSTM)**:
   - I built a Long Short-Term Memory (**LSTM**) neural network using **PyTorch**.
   - The LSTM takes sliding windows of the past 10 days of data, including the Stock Close Price, Volume, and the extracted FinBERT Sentiment Score.
   - It learns the complex, non-linear relationships between market sentiment and price action to predict the next day's stock price.

## 📂 Project Structure

- `scripts/1_collect_data.py`: Downloads AAPL stock data using `yfinance` and generates our financial news dataset.
- `scripts/2_analyze_sentiment.py`: Runs the FinBERT Neural Network to extract sentiment scores.
- `scripts/3_prepare_time_series.py`: Fuses the data, scales it (MinMaxScaler), and generates sequential windows for PyTorch.
- `scripts/4_train_lstm.py`: Defines and trains the PyTorch LSTM model.
- `scripts/5_evaluate_model.py`: Runs predictions on unseen test data and generates visualizations.
- `run_pipeline.sh`: A single script to execute the entire pipeline end-to-end.

## 📉 Results

By combining textual sentiment with numerical data, the LSTM successfully captures market trends. You can find the AI's predicted vs. actual stock price graph in the `reports/figures/` directory!

---
*Created by Farhan Ahmad Ansari.*
