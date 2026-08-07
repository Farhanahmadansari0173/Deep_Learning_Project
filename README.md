# 📈 Multimodal Deep Learning for Financial Forecasting (FinBERT + LSTM)

Welcome to my Deep Learning project! My name is **Farhan Ahmad Ansari**, and I built this repository to explore how Natural Language Processing (NLP) can be fused with Time-Series Forecasting to predict stock market movements.

Standard algorithmic trading models rely purely on historical numbers (like Open, Close, and Volume). However, the market is heavily driven by human emotion and news. In this project, I built an AI pipeline that actually *reads* financial news, understands the sentiment, and uses that context alongside historical prices to make highly accurate predictions.

## 🧠 The Architecture

I designed this project using a dual-pipeline approach to process both text and numbers simultaneously:

1. **The NLP Pipeline (Transformers)**:
   - Traditional NLP models don't understand financial terminology (e.g., "short" usually implies a duration or size, but in finance, it means betting against a stock). 
   - To solve this, I implemented **FinBERT** (a HuggingFace Transformer pre-trained on financial text) to analyze financial news headlines.
   - The model extracts a daily "Sentiment Score" (Positive/Negative/Neutral) based on market news.

2. **The Time-Series Pipeline (LSTM)**:
   - I built a Long Short-Term Memory (**LSTM**) neural network from scratch using **PyTorch**.
   - The LSTM takes sliding windows of the past 10 days of data, including the Stock Close Price, Volume, and the extracted FinBERT Sentiment Score.
   - By feeding it sequential data, the network learns the complex, non-linear relationships between market sentiment and price action to predict the next day's stock price.

## 📂 Project Structure

I separated the codebase into modular scripts to handle the entire lifecycle of the data:

- `scripts/1_collect_data.py`: Downloads AAPL stock data using `yfinance` and generates the financial news dataset.
- `scripts/2_analyze_sentiment.py`: Runs the FinBERT Neural Network to extract sentiment scores.
- `scripts/3_prepare_time_series.py`: Fuses the data, scales it (MinMaxScaler), and generates sequential windows for PyTorch.
- `scripts/4_train_lstm.py`: Defines and trains the PyTorch LSTM model.
- `scripts/5_evaluate_model.py`: Runs predictions on unseen test data and calculates RMSE, MAE, and MAPE.
- `scripts/6_extra_visualizations.py`: Generates correlation heatmaps and sentiment distribution pie charts.
- `run_pipeline.sh`: A single script to execute my entire pipeline end-to-end.

## 📉 Results & Evaluation

By combining textual sentiment with numerical data, my LSTM successfully captured the market trends with high precision. 

On the unseen test dataset, the model achieved:
*   **RMSE (Root Mean Squared Error):** $3.29
*   **MAE (Mean Absolute Error):** $2.64
*   **MAPE (Mean Absolute Percentage Error):** 1.47%

A MAPE of **1.47%** indicates that the model's predictions deviate by an average of only 1.47% from the true future stock price.

You can view the AI's predicted vs. actual stock price graph, as well as the correlation heatmaps, in the `reports/figures/` directory!

---
*Created by Farhan Ahmad Ansari.*
