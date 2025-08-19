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

# Positional encoding for transformer models to add temporal information to input data
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):  
        # Add positional encoding to input tensor
        T = x.size(1)
        return x + self.pe[:, :T, :]


# Attention-based pooling mechanism to focus on important time steps
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Initialize attention layer
        self.attn = nn.Linear(d_model, 1)  

    def forward(self, h):  
        # Compute attention scores and weighted sum of time steps
        scores = self.attn(h)
        weights = torch.softmax(scores, dim=1)
        
        context = torch.sum(weights * h, dim=1) 
        return context


# Transformer-based regression model for time series data
class TransformerRegressor(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1,
                 pool: str = 'mean'):
        super().__init__()
        # Initialize transformer encoder and pooling mechanism
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool_type = pool
        if pool == 'attention':
            self.attn_pool = AttentionPooling(d_model)

        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, x):
        # Forward pass through transformer encoder and pooling mechanism
        h = self.input_proj(x)  
        h = self.pos_enc(h)
        h = self.encoder(h)

        if self.pool_type == 'mean':
            h = h.mean(dim=1)
        elif self.pool_type == 'last':
            h = h[:, -1, :]
        elif self.pool_type == 'attention':
            h = self.attn_pool(h)
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")

        out = self.head(h)
        return out

# Dataset class for handling sequences of features and targets
class SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        # Convert input sequences to PyTorch tensors
        self.X = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y_seq).float()
    def __len__(self):
        # Return the number of sequences in the dataset
        return len(self.X)
    def __getitem__(self, idx):
        # Retrieve a single sequence and its target
        return self.X[idx], self.y[idx]

# Function to create sequences of features and targets for time series data
def make_sequences(features: np.ndarray, targets: np.ndarray, time_steps: int = 8):
    # Generate sequences of specified time steps
    Xs, ys = [], []
    for i in range(len(features) - time_steps):
        Xs.append(features[i:i + time_steps, :])
        ys.append(targets[i + time_steps, :])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

# Function to train the transformer model on time series data
def train_transformer(feat_array, targ_array, returns_scaler):
    # Prepare sequences and split into training and testing sets
    TIME_STEPS = 16
    X_seq, y_seq = make_sequences(feat_array, targ_array, time_steps=TIME_STEPS)

    split_idx = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    train_ds = SequenceDataset(X_train, y_train)
    test_ds = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]
    model = TransformerRegressor(
        n_features=n_features,
        n_outputs=n_outputs,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        pool='attention'
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    patience = 10
    stale_epochs = 0
    max_epochs = 100

    val_split = max(1, int(0.1 * len(train_ds)))
    train_core = SequenceDataset(X_train[:-val_split], y_train[:-val_split]) if len(train_ds) > 10 else train_ds
    val_core = SequenceDataset(X_train[-val_split:], y_train[-val_split:]) if len(train_ds) > 10 else test_ds

    train_loader = DataLoader(train_core, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_core, batch_size=64, shuffle=False)

    # Training loop with early stopping based on validation loss
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

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if 'best_state' in locals():
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

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

    print("Model Evaluation (PyTorch Transformer):")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Directional Accuracy: {directional_accuracy:.2%}")

    last_window = feat_array[-TIME_STEPS:, :].reshape(1, TIME_STEPS, feat_array.shape[1]).astype(np.float32)
    with torch.no_grad():
        next_scaled_returns = model(torch.from_numpy(last_window).to(device)).cpu().numpy()[0]

    return next_scaled_returns