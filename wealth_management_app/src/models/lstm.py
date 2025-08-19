import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import os

# Set random seed for reproducibility across libraries and environments
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# ---------- PyTorch Dataset for sequences ----------
class SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        self.X = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y_seq).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------- LSTM Model ----------
class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, hidden_sizes=(64, 32), dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_sizes[1], n_outputs)

    def forward(self, x):  # x: (batch, time, features)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        # take last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# ---------- Utility: make sequences ----------
def make_sequences(features: np.ndarray, targets: np.ndarray, time_steps: int = 8):
    Xs, ys = [], []
    for i in range(len(features) - time_steps):
        Xs.append(features[i:i + time_steps, :])
        ys.append(targets[i + time_steps, :])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm(feat_array, targ_array, returns_scaler):

    # Build sequences for LSTM
    TIME_STEPS = 16
    X_seq, y_seq = make_sequences(feat_array, targ_array, time_steps=TIME_STEPS)

    # Time-based split
    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # Dataloaders
    train_ds = SequenceDataset(X_train, y_train)
    test_ds = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]
    model = LSTMRegressor(n_features=n_features, n_outputs=n_outputs, hidden_sizes=(64, 32), dropout=0.2).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Early stopping settings
    best_val_loss = float('inf')
    patience = 10
    stale_epochs = 0
    max_epochs = 100

    # Split a small validation slice from the end of train
    val_split = max(1, int(0.1 * len(train_ds)))
    train_core = SequenceDataset(X_train[:-val_split], y_train[:-val_split]) if len(train_ds) > 10 else train_ds
    val_core = SequenceDataset(X_train[-val_split:], y_train[-val_split:]) if len(train_ds) > 10 else test_ds

    train_loader = DataLoader(train_core, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_core, batch_size=64, shuffle=False)

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        # Early stopping check
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    # Restore best model
    if 'best_state' in locals():
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Evaluate on test set
    model.eval()
    preds_list, truth_list = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            preds_list.append(preds)
            truth_list.append(yb.numpy())

    y_pred = np.vstack(preds_list)
    y_true = np.vstack(truth_list)

    y_pred_unscaled = returns_scaler.inverse_transform(y_pred)
    y_true_unscaled = returns_scaler.inverse_transform(y_true)

    mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
    correct_directions = np.sum(np.sign(y_pred_unscaled) == np.sign(y_true_unscaled))
    total_directions = y_true_unscaled.size
    directional_accuracy = correct_directions / total_directions

    print("Model Evaluation (PyTorch LSTM):")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Directional Accuracy: {directional_accuracy:.2%}")

    # Predict next step (scaled returns)
    last_window = feat_array[-TIME_STEPS:, :].reshape(1, TIME_STEPS, feat_array.shape[1]).astype(np.float32)
    with torch.no_grad():
        next_scaled_returns = model(torch.from_numpy(last_window).to(device)).cpu().numpy()[0]

    return next_scaled_returns
