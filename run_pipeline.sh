#!/bin/bash
set -e

echo "=== Deep Learning Project Pipeline ==="
echo "1. Creating Python Virtual Environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "2. Installing requirements..."
pip install -r requirements.txt

echo "3. Collecting Stock & News Data..."
python scripts/1_collect_data.py

echo "4. Running FinBERT Sentiment Analysis..."
python scripts/2_analyze_sentiment.py

echo "5. Preparing Time-Series Windows..."
python scripts/3_prepare_time_series.py

echo "6. Training the LSTM Model..."
python scripts/4_train_lstm.py

echo "7. Evaluating and Plotting..."
python scripts/5_evaluate_model.py

echo "=== Pipeline Completed Successfully! ==="
