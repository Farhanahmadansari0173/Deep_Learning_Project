"""
Script 4: Training the Long Short-Term Memory (LSTM) Model

Now for the Deep Learning! We define a PyTorch LSTM network. 
Unlike standard neural networks that treat every input independently, 
LSTMs possess an internal "memory" state, allowing them to understand trends 
over time (e.g., if a stock goes up for 3 days alongside positive news, it learns that pattern).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

# Hyperparameters
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

class StockLSTM(nn.Module):
    """
    A standard PyTorch LSTM architecture.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The LSTM layer (takes our sequence of [Close, Volume, Sentiment])
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # A fully connected linear layer to condense the LSTM's final state into a single price prediction
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate through the LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # We only care about the output of the very last time step to make our prediction
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    print("Loading prepared sequences...")
    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    
    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # [batch_size, 1]
    
    # Create DataLoaders for efficient batch processing
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    input_size = X_train.shape[2] # Number of features (e.g., 3: Close, Volume, Sentiment)
    
    print("Initializing LSTM Model...")
    model = StockLSTM(input_size=input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss() # Mean Squared Error is standard for regression/forecasting
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.6f}")
            
    print("Training complete!")
    
    # Save the trained model parameters
    model_path = os.path.join(MODELS_DIR, "lstm_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
