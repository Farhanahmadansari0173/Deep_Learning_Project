"""
Script 5: Evaluating the Model and Visualizing Predictions

We run our trained LSTM on the unseen Test data to see how well it learned the patterns.
We then use matplotlib to generate a beautiful line chart comparing the AI's 
predicted stock price against the actual stock price.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Define the model architecture again to load the weights
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading test data...")
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    print("Loading trained LSTM model...")
    input_size = X_test.shape[2]
    model = StockLSTM(input_size=input_size, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm_model.pth")))
    model = model.to(device)
    model.eval()
    
    print("Generating predictions on the test set...")
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
        
    # Un-scale the data to get real dollar amounts back!
    print("Inverting MinMaxScaler to calculate real dollar amounts...")
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
        
    # We need to recreate a dummy array of shape (N, features) to unscale the predictions
    # since our scaler was fit on [Close, Volume, Sentiment]
    num_features = scaler.scale_.shape[0]
    
    dummy_pred = np.zeros((len(predictions), num_features))
    dummy_pred[:, 0] = predictions[:, 0]
    real_predictions = scaler.inverse_transform(dummy_pred)[:, 0]
    
    dummy_y = np.zeros((len(y_test), num_features))
    dummy_y[:, 0] = y_test
    real_y_test = scaler.inverse_transform(dummy_y)[:, 0]
    
    rmse = np.sqrt(mean_squared_error(real_y_test, real_predictions))
    mae = mean_absolute_error(real_y_test, real_predictions)
    mape = np.mean(np.abs((real_y_test - real_predictions) / real_y_test)) * 100
    
    print(f"\nFinal Test RMSE (Root Mean Squared Error): ${rmse:.2f}")
    print(f"Final Test MAE (Mean Absolute Error): ${mae:.2f}")
    print(f"Final Test MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    
    # Plotting
    print("Generating prediction visualization...")
    plt.figure(figsize=(14, 6))
    plt.plot(real_y_test, color='black', label='Actual Stock Price')
    plt.plot(real_predictions, color='green', label='AI Predicted Stock Price')
    plt.title('Stock Price Prediction using FinBERT + LSTM')
    plt.xlabel('Time (Days in Test Set)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(FIGURES_DIR, "stock_prediction.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Visualization saved successfully to {plot_path}")

if __name__ == "__main__":
    main()
