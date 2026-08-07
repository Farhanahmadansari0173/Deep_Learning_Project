"""
Script 2: Sentiment Analysis using FinBERT

Traditional NLP models don't understand finance (e.g., "short" is just an adjective to BERT, 
but to a trader, it means betting against a stock).
Here, we use FinBERT, a transformer model pre-trained specifically on financial text.
We pass all of our daily headlines through this Neural Network to extract a "Sentiment Score".
"""
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def analyze_sentiment(news_df):
    print("Loading FinBERT Model (This will download ~400MB if not cached)...")
    # ProsusAI/finbert is the industry standard for financial sentiment
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    # Check if GPU is available (MPS for Mac, CUDA for Nvidia)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # Mapping FinBERT's output labels to numerical scores
    # Typically: 0=Positive, 1=Negative, 2=Neutral (Check model config for exact mapping)
    # Actually for ProsusAI/finbert: 0=positive, 1=negative, 2=neutral
    sentiment_map = {0: 1.0, 1: -1.0, 2: 0.0} 
    
    print(f"Analyzing sentiment for {len(news_df)} headlines...")
    
    sentiment_scores = []
    
    # Process in batches for speed
    batch_size = 32
    headlines = news_df['Headline'].tolist()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(headlines), batch_size), desc="FinBERT"):
            batch = headlines[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            # Get the predicted class for each headline in the batch
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(predictions, dim=-1).cpu().numpy()
            
            for cls in predicted_classes:
                sentiment_scores.append(sentiment_map[cls])
                
    news_df['Sentiment'] = sentiment_scores
    return news_df

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("Loading raw financial news...")
    news_path = os.path.join(DATA_DIR, "financial_news.csv")
    news_df = pd.read_csv(news_path)
    
    # Run the neural network!
    scored_news_df = analyze_sentiment(news_df)
    
    print("Aggregating sentiment by day...")
    # Group by Date and calculate the average sentiment score for that day
    daily_sentiment = scored_news_df.groupby('Date')['Sentiment'].mean().reset_index()
    
    output_path = os.path.join(PROCESSED_DIR, "daily_sentiment.csv")
    daily_sentiment.to_csv(output_path, index=False)
    
    print(f"\nSentiment Analysis complete! Saved daily scores to {output_path}")

if __name__ == "__main__":
    main()
