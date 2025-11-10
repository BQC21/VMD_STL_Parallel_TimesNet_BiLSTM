
##################################
#### Packages ###################
##################################

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from time import perf_counter
from tqdm.auto import tqdm, trange

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

# Processing
def build_loaders_for_imf(df, imf_col = None,
                          seq_len=48, pred_len=1, batch=64):

    """
    Create Dataloaders for training, validation and testing for a given IMF column.
    """
    n = len(df)
    n_train = int(round(0.6 * n))
    n_val = int(round(0.8 * n))
    print(f"Dataset size: train={n_train} | val={n-n_val} | test={n - (n_val+n_train)}")

    target_col = f"Active_Power_{imf_col}"
    features = [c for c in df.columns if c != target_col]

    FEATURES_train = df[features][:n_train]
    FEATURES_valid = df[features][n_train:n_val]
    FEATURES_test = df[features][n_val:]

    scaler_x = StandardScaler()
    scaler_y = MinMaxScaler() 

    # Convert to real arrays if they come as complex (VMD can return complex dtype
    # even when the imaginary part is zero). Scikit-learn does not accept complex data.
    def _to_real_array(df_part):
        arr = df_part.values
        if np.iscomplexobj(arr):
            warnings.warn("Complex values found in input -- discarding imaginary part and using real part.")
            arr = np.real(arr)
        return arr.astype(np.float64)

    X_train_arr = _to_real_array(FEATURES_train)
    X_val_arr = _to_real_array(FEATURES_valid)
    X_test_arr = _to_real_array(FEATURES_test)

    y_train_arr = df[[target_col]].iloc[:n_train].values
    if np.iscomplexobj(y_train_arr):
        warnings.warn("Complex values found in target -- discarding imaginary part and using real part.")
        y_train_arr = np.real(y_train_arr)
    y_train_arr = y_train_arr.astype(np.float64)

    y_val_arr = df[[target_col]].iloc[n_train:n_val].values
    if np.iscomplexobj(y_val_arr):
        y_val_arr = np.real(y_val_arr)
    y_val_arr = y_val_arr.astype(np.float64)

    y_test_arr = df[[target_col]].iloc[n_val:].values
    if np.iscomplexobj(y_test_arr):
        y_test_arr = np.real(y_test_arr)
    y_test_arr = y_test_arr.astype(np.float64)

    X_train = scaler_x.fit_transform(X_train_arr)
    y_train = scaler_y.fit_transform(y_train_arr)

    X_val = scaler_x.transform(X_val_arr)
    y_val = scaler_y.transform(y_val_arr)

    X_test = scaler_x.transform(X_test_arr)
    y_test = scaler_y.transform(y_test_arr)

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
                 epochs=30, patience=10, verbose=True):

    from numpy import mean

    loss_train, loss_valid = [], []
    best_val = np.inf
    wait = 0
    epoch_times, val_times = [], []
    t0_total = perf_counter()

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

        train_mse = float(mean(batch_losses))
        loss_train.append(train_mse)

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
            tqdm.write(f"Epoch {epoch:03d} | train_mse={train_mse:.6f} | val_mse={vloss:.6f} | t_epoch={epoch_time:.2f}s | t_val={val_time:.2f}s")

        # Early stopping
        if vloss < best_val:
            best_val, wait = vloss, 0
            torch.save(model.state_dict(), MODEL_PATH)
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

    return loss_train, loss_valid, stats_tiempo