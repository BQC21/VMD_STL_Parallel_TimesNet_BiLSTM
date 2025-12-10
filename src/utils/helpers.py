
##################################
#### Packages ###################
##################################
import os, sys, warnings, yaml, random
warnings.filterwarnings("ignore")

# Make imports work whether this file is run as a script or as a module.
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))        # .../src
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))  # repository root
for _p in (SRC_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# NumPy / Pandas
import numpy as np
import pandas as pd

# Torch
import torch
from torch.utils.data import Dataset, DataLoader

# sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

# Utils 
from pipeline.metrics import compute_metrics

# Computational time
from time import perf_counter
from tqdm import tqdm


##################################

## Sequences
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_sequences(X, y, seq_len=48, pred_len=1):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_len + 1):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len+pred_len-1])
    return np.array(Xs), np.array(ys)

# prediction
def predict_loader(model, dl, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                out = model(xb)
            preds.append(out.detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

# Processing
def build_loaders(df, seq_len=48, pred_len=1, batch=64, target_col="Active_Power"):

    """
    Create Dataloaders for training, validation and testing
    """
    n = len(df)
    n_train = int(round(0.6 * n))
    n_val = int(round(0.8 * n))
    print(f"Dataset size: train={n_train} | val={n_val - n_train} | test={n - n_val}")

    train_df = df[:n_train]
    val_df = df[n_train:n_val]
    test_df = df[n_val:]

    features = [c for c in train_df.columns if c != target_col]

    scaler_x = StandardScaler()
    scaler_y = MinMaxScaler() 

    X_train = scaler_x.fit_transform(train_df[features])
    y_train = scaler_y.fit_transform(train_df[[target_col]])
    X_val = scaler_x.transform(val_df[features])
    y_val = scaler_y.transform(val_df[[target_col]])
    X_test = scaler_x.transform(test_df[features])
    y_test = scaler_y.transform(test_df[[target_col]])

    Xtr, ytr = make_sequences(X_train, y_train, seq_len=seq_len, pred_len=pred_len)
    Xva, yva = make_sequences(X_val,   y_val,   seq_len=seq_len, pred_len=pred_len)
    Xts, yts = make_sequences(X_test,   y_test,   seq_len=seq_len, pred_len=pred_len)

    train_dl = DataLoader(SeqDataset(Xtr, ytr), batch_size=batch, shuffle=True, drop_last=True, pin_memory=True)
    val_dl   = DataLoader(SeqDataset(Xva, yva), batch_size=batch, shuffle=False, pin_memory=True)
    test_dl  = DataLoader(SeqDataset(Xts, yts), batch_size=batch, shuffle=False, pin_memory=True)

    return train_dl, val_dl, test_dl, scaler_y, len(features)  

#####################################
# Evaluate (for test and validation)
#####################################

def evaluate(model, dl, device="cuda", use_amp=True):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if use_amp and "cuda" in device:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(xb)
            else:
                out = model(xb)

            if out.shape != yb.shape:
                raise RuntimeError(f"Shape mismatch: {out.shape} vs {yb.shape}")

            preds.append(out.detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())

    preds = np.concatenate(preds).reshape(-1, 1)
    trues = np.concatenate(trues).reshape(-1, 1)
    return preds, trues

#####################################
# Training with AMP 
#####################################

def training_amp(model, device, loss_fn, scaler, optim,
                train_dl, val_dl, MODEL_PATH, 
                df, seq_len, pred_len,    
                epochs=30, patience=10, verbose=True):

    from numpy import mean

    true_val = np.zeros(int(0.2*len(df)-seq_len-pred_len+1))  
    pred_val = np.zeros(int(0.2*len(df)-seq_len-pred_len+1))  
    # true_val = np.zeros(6893) # Figshare
    # pred_val = np.zeros(6893) # Figshare
    
    loss_train, loss_valid = [], []
    best_val = np.inf
    wait = 0
    
    epoch_times, val_times = [], []
    t0_total = perf_counter()

    from tqdm import trange  # fix: import trange from tqdm

    for epoch in trange(epochs, desc="Training (epochs)", dynamic_ncols=True):
        model.train()
        batch_losses = []
        t0_epoch = perf_counter()

        for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch:03d} [train]", leave=False, dynamic_ncols=True):
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)

            use_amp = "cuda" in device
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(xb)
                    if out.shape != yb.shape:
                        raise RuntimeError(f"Shape mismatch {out.shape} vs {yb.shape}")
                    loss = loss_fn(out, yb)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                out = model(xb)
                if out.shape != yb.shape:
                    raise RuntimeError(f"Shape mismatch {out.shape} vs {yb.shape}")
                loss = loss_fn(out, yb)
                loss.backward()
                optim.step()

            batch_losses.append(loss.item())

        epoch_time = perf_counter() - t0_epoch
        epoch_times.append(epoch_time)

        loss_train.append(np.mean(batch_losses))

        # Validation
        t0_val = perf_counter()
        model.eval()
        with torch.no_grad():
            vp, vt = evaluate(model, val_dl, device=device, use_amp=("cuda" in device))
        vloss = float(mean_squared_error(vp, vt))
        val_time = perf_counter() - t0_val
        val_times.append(val_time)
        loss_valid.append(vloss)

        if verbose:
            tqdm.write(f"Epoch {epoch:03d} | train_mse={np.mean(batch_losses):.6f} | val_mse={vloss:.6f} | t_epoch={epoch_time:.2f}s | t_val={val_time:.2f}s")

        # Early stopping
        if vloss < best_val:
            best_val, wait = vloss, 0
            torch.save(model.state_dict(), MODEL_PATH)
            true_val, pred_val = vt, vp # update best valid predictions
        else:
            wait += 1
            if wait >= patience:
                tqdm.write("Early stopping.")
                break

        if torch.cuda.is_available() and "cuda" in device:
            torch.cuda.empty_cache()

    total_time = perf_counter() - t0_total
    stats_tiempo = {
        "total_s": total_time,
        "n_epochs": len(loss_train),
        "epoch_avg_s": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "val_avg_s": float(np.mean(val_times)) if val_times else 0.0,
    }

    if verbose:
        tqdm.write(f"[TIMES] total={total_time:.2f}s | epoch_avg={stats_tiempo['epoch_avg_s']:.2f}s | val_avg={stats_tiempo['val_avg_s']:.2f}s")

    return loss_train, loss_valid, stats_tiempo, true_val, pred_val

#####################################
# train whole model
#####################################
def train_whole_model(df, model, DEVICE, CKPT_DIR, 
                    SEQ_LEN, PRED_LEN, model_name_file,
                    BATCH_SIZE, TARGET, 
                    LR, PATIENCE_ES, loss_fn, scaler):
    # Load sequences
    train_dl, val_dl, test_dl, y_scaler, __ = build_loaders(
        df=df, seq_len=SEQ_LEN, pred_len=PRED_LEN, 
        batch=BATCH_SIZE, target_col=TARGET
    )

    # -------- Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Use a filename that doesn't depend on VMD-specific variables when VMD is disabled
    model_path = os.path.join(CKPT_DIR, model_name_file)

    # Training with AMP + early stopping
    __, __, __, vt, vp = training_amp(
        model=model, device=str(DEVICE), 
        loss_fn=loss_fn, scaler=scaler,
        optim=optim, train_dl=train_dl, val_dl=val_dl,
        MODEL_PATH=model_path, df = df, 
        seq_len=SEQ_LEN, pred_len=PRED_LEN,
        patience=PATIENCE_ES, verbose=True
    )
    y_pred_inv = y_scaler.inverse_transform(vp.reshape(-1, 1)).ravel()
    y_true_inv = y_scaler.inverse_transform(vt.reshape(-1, 1)).ravel()
    print(f"shapes of true and predicted values: {y_true_inv.shape}, {y_pred_inv.shape}")

    return y_true_inv, y_pred_inv, model_path, test_dl, y_scaler

#####################################
# plot metrics
#####################################
def print_metrics(Y_real_total, Y_pred_total):

    # Compute metrics (RMSE) after all IMFs
    R2, MAE, RMSE = compute_metrics(Y_real_total, Y_pred_total)
    tqdm.write(f"\n=== VMD Reconstruction Results ===")
    tqdm.write(f"RMSE: {RMSE:.4f}")
    tqdm.write(f"MAE:  {MAE:.4f}")
    tqdm.write(f"R2:   {R2:.4f}")    