"""
Script 6: Extra Academic Visualizations

To make the thesis outstanding, we generate extra figures that prove our methodology:
1. Sentiment Distribution Plot (Pie Chart)
2. Correlation Heatmap (Close Price vs Volume vs Sentiment)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
FIGURES_DIR = "reports/figures"

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # 1. Sentiment Distribution
    print("Generating Sentiment Distribution Plot...")
    sentiment_df = pd.read_csv(os.path.join(PROCESSED_DIR, "daily_sentiment.csv"))
    
    plt.figure(figsize=(8, 6))
    # Categorize sentiment back into text for visualization
    def categorize(score):
        if score > 0.3: return 'Positive'
        elif score < -0.3: return 'Negative'
        else: return 'Neutral'
        
    sentiment_df['Category'] = sentiment_df['Sentiment'].apply(categorize)
    category_counts = sentiment_df['Category'].value_counts()
    
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, colors=['#4CAF50', '#FFC107', '#F44336'])
    plt.title('Distribution of Financial News Sentiment')
    
    pie_path = os.path.join(FIGURES_DIR, "sentiment_distribution.png")
    plt.savefig(pie_path, dpi=300)
    plt.close()
    
    # 2. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    stock_df = pd.read_csv(os.path.join(DATA_DIR, "stock_prices.csv"))
    merged_df = pd.merge(stock_df, sentiment_df, on='Date', how='left').fillna(0.0)
    
    features = ['Close', 'Volume', 'Sentiment']
    corr_matrix = merged_df[features].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title('Correlation Matrix (Price vs Sentiment)')
    
    heatmap_path = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    print(f"Extra visualizations saved to {FIGURES_DIR}")

if __name__ == "__main__":
    main()
