''' ---------------------------------------------------------------------------------------------------------------------------------------------------------
Author: Elijah Boon
Created: 2025-08-17
Description: AI-driven anti-drowsiness detection system (LSTM-based).
--------------------------------------------------------------------------------------------------------------------------------------------------------------'''
import numpy as np
import pandas as pd
from LuminaLSTM import DrowsinessLSTM
import torch
import torch.nn as nn
import torch.optim as op
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm


FILE = 'combined_with_transitions.csv'

input_size = 5
hidden_size = 128
num_layers = 2
output_size = 1   # Binary classification


def data_extraction(FILE, seq_len=30):
    df = pd.read_csv(FILE)
    
    df['ear_smooth'] = df['EAR'].rolling(window=5, min_periods=1).mean()
    df['mar_smooth'] = df['MAR'].rolling(window=5, min_periods=1).mean()
    df['perclos_smooth'] = df['PERCLOS'].rolling(window=30, min_periods=1).mean()
    df['yawn_smooth'] = df['YAWNS'].rolling(window=30, min_periods=1).mean()
    df['pitch_smooth'] = df['Pitch'].rolling(window=5, min_periods=1).mean()

    # Fixed: removed missing column
    X = df[['ear_smooth', 'mar_smooth', 'perclos_smooth', 'yawn_smooth', 'pitch_smooth']].values
    y = df['label'].astype(float).values


    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def create_sequences(X, y, seq_len):
        sequences, labels = [], []
        for i in range(len(X) - seq_len + 1):
            seq = X[i: i + seq_len]
            label = y[i + seq_len - 1]
            sequences.append(seq)
            labels.append(label)
        return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq


def train_model(X, y, num_epochs=5, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert dataset to tensors
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)  # BCE loss needs float

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = DrowsinessLSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification
    optimizer = op.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, leave=False):  
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).squeeze()  # [batch]
            loss = criterion(outputs, labels.float())  # BCE loss expects float labels

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()  # ✅ accumulate per batch

        avg_loss = epoch_loss / len(train_loader)  
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(targets.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    torch.save(model.state_dict(), "drowsiness_lstm_binary31.pth")
    print("[INFO] PyTorch binary model saved.")


if __name__ == "__main__":
    X_train_seq, y_train_seq, X_test_seq, y_test_seq = data_extraction(FILE)
    train_model(X_train_seq, y_train_seq)
