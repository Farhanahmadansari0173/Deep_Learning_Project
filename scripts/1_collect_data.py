"""
Script 1: Data Collection (Stock Prices & Financial News)

In this script, I collect the two datasets I need for my Multimodal Deep Learning pipeline:
1. Historical Stock Prices (using yfinance)
2. Financial News Headlines

Since paid financial news APIs are expensive, I simulate a daily news feed by pulling 
realistic financial headlines and mapping them to trading days. This ensures my 
project is fully reproducible.
"""
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datasets import load_dataset
from datetime import datetime, timedelta

DATA_DIR = "data/raw"

def get_stock_data(ticker="AAPL", start_date="2022-01-01", end_date="2024-01-01"):
    print(f"Downloading historical stock data for {ticker}...")
    stock = yf.download(ticker, start=start_date, end=end_date)
    
    # yfinance sometimes returns a MultiIndex column structure in newer versions. 
    # We flatten it to simple column names.
    if isinstance(stock.columns, pd.MultiIndex):
        stock.columns = [col[0] for col in stock.columns]
        
    stock.reset_index(inplace=True)
    stock['Date'] = pd.to_datetime(stock['Date']).dt.date
    print(f"Downloaded {len(stock)} days of stock data.")
    return stock

def get_financial_news(dates):
    print("Generating simulated financial news headlines...")
    
    # Realistic financial phrases from the phrasebank domain
    sentences = [
        "The company reported a massive surge in quarterly profits.",
        "Revenue fell short of Wall Street expectations causing the stock to plummet.",
        "A new product launch has generated significant positive momentum.",
        "The CEO resigned unexpectedly, causing market uncertainty.",
        "Operating margins improved by 5% year over year.",
        "Supply chain disruptions have caused a delay in production.",
        "The firm announced a 2-for-1 stock split.",
        "Analysts upgraded the stock to a Strong Buy.",
        "The federal reserve announced a rate hike which negatively impacted the sector.",
        "The company beat earnings per share estimates by a wide margin.",
        "Regulatory hurdles have delayed the merger.",
        "Record breaking sales were reported for the holiday quarter."
    ]
    
    print(f"Loaded {len(sentences)} simulated financial news headlines.")
    
    # To simulate a real news feed, we will randomly assign 1-3 headlines to every trading day
    news_records = []
    
    # Set a random seed for reproducibility
    np.random.seed(42)
    
    for date in dates:
        num_articles = np.random.randint(1, 4)
        daily_sentences = np.random.choice(sentences, size=num_articles, replace=False)
        for sentence in daily_sentences:
            news_records.append({
                "Date": date,
                "Headline": sentence
            })
            
    news_df = pd.DataFrame(news_records)
    print(f"Generated a simulated daily news feed with {len(news_df)} total headlines.")
    return news_df

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Get Stock Data
    stock_df = get_stock_data(ticker="AAPL", start_date="2022-01-01", end_date="2024-01-01")
    stock_path = os.path.join(DATA_DIR, "stock_prices.csv")
    stock_df.to_csv(stock_path, index=False)
    
    # 2. Get News Data mapped to the exact trading days
    trading_dates = stock_df['Date'].unique()
    news_df = get_financial_news(trading_dates)
    news_path = os.path.join(DATA_DIR, "financial_news.csv")
    news_df.to_csv(news_path, index=False)
    
    print(f"\nData collection complete! Saved to {DATA_DIR}")

if __name__ == "__main__":
    main()
