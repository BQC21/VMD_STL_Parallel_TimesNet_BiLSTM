
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

# Utils 
from pipeline.metrics import compute_metrics
from visualization.plots import visualize, scatter

# DL model
from models.TimesNet_BiLSTM import TimesNet_BiLSTM_Parallel, BiLSTM, TimesNet

# STL
from statsmodels.tsa.seasonal import STL

# VMD
from features.vmd import VMD

# Computational time
from time import perf_counter
from types import SimpleNamespace
from typing import Any, cast
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

def build_loaders_for_imf(df, imf_col = None,
                        seq_len=48, pred_len=1, 
                        batch=64, target_col="Active_Power"):

    """
    Create Dataloaders for training, validation and testing for a given IMF column.
    """
    n = len(df)
    n_train = int(round(0.6 * n))
    n_val = int(round(0.8 * n))
    print(f"Dataset size: train={n_train} | val={n_val - n_train} | test={n - n_val}")

    train_df = df[:n_train]
    val_df = df[n_train:n_val]
    test_df = df[n_val:]

    target_col_imf = f"{target_col}_{imf_col}"
    features = [c for c in train_df.columns if c != target_col_imf] 

    FEATURES_train = train_df[features]
    FEATURES_valid = val_df[features]
    FEATURES_test = test_df[features]

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

    y_train_arr = train_df[target_col_imf].values
    if np.iscomplexobj(y_train_arr):
        warnings.warn("Complex values found in target -- discarding imaginary part and using real part.")
        y_train_arr = np.real(y_train_arr)
    y_train_arr = y_train_arr.astype(np.float64).reshape(-1, 1)

    y_val_arr = val_df[target_col_imf].values
    if np.iscomplexobj(y_val_arr):
        y_val_arr = np.real(y_val_arr)
    y_val_arr = y_val_arr.astype(np.float64).reshape(-1, 1)

    y_test_arr = test_df[target_col_imf].values
    if np.iscomplexobj(y_test_arr):
        y_test_arr = np.real(y_test_arr)
    y_test_arr = y_test_arr.astype(np.float64).reshape(-1, 1)

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
# train for each IMF
#####################################
def train_for_each_imf(df, model, u_all, SIGNALS, 
                    SIGNAL_NAMES, DEVICE, CKPT_DIR, 
                    SEQ_LEN, PRED_LEN, 
                    BATCH_SIZE, TARGET, 
                    K, Y_pred_total, Y_real_total,
                    LR, EPOCHS, PATIENCE_ES, loss_fn, scaler):
    models = []

    for idx in range(K):
        imf_col = f"mode{idx}"

        # build dataframe for this IMF
        modes_df = pd.DataFrame({
            f"{SIGNAL_NAMES[sig_idx]}_{imf_col}": u_all[sig_idx][idx, :]
            for sig_idx in range(len(SIGNALS))
        })

        tqdm.write(f"\n=== Training {imf_col} ===")

        # Load sequences
        train_dl, val_dl, test_dl, y_scaler, __ = build_loaders_for_imf(
            df=modes_df, imf_col=imf_col,
            seq_len=SEQ_LEN, pred_len=PRED_LEN, 
            batch=BATCH_SIZE, target_col=TARGET
        )

        # --------- Model + optimizer
        model_i = model
        optim_i = torch.optim.Adam(model_i.parameters(), lr=LR, weight_decay=1e-4)
        model_path = os.path.join(CKPT_DIR, f"model_imf_{idx}.pt")

        # Training with AMP + early stopping
        __, __, stats_t, vt, vp = training_amp(
            model=model_i, device=str(DEVICE), 
            loss_fn=loss_fn, scaler=scaler,
            optim=optim_i, train_dl=train_dl, val_dl=val_dl,
            MODEL_PATH=model_path, df = df, 
            seq_len=SEQ_LEN, pred_len=PRED_LEN,
            epochs=EPOCHS, patience=PATIENCE_ES, verbose=True
        )

        models.append(model_i)

        # Ensure vp and vt are numpy arrays with shape (n_samples, 1) for inverse_transform
        vp_arr = np.asarray(vp)
        vt_arr = np.asarray(vt)
        if vp_arr.ndim == 1:
            vp_arr = vp_arr.reshape(-1, 1)
        else:
            vp_arr = vp_arr.reshape(vp_arr.shape[0], -1)
        if vt_arr.ndim == 1:
            vt_arr = vt_arr.reshape(-1, 1)
        else:
            vt_arr = vt_arr.reshape(vt_arr.shape[0], -1)
    
        y_pred_inv = y_scaler.inverse_transform(vp_arr).ravel()
        y_true_inv = y_scaler.inverse_transform(vt_arr).ravel()
        print(f"shapes of true and predicted values: {y_true_inv.shape}, {y_pred_inv.shape}")

        Y_pred_total += y_pred_inv.flatten()
        Y_real_total += y_true_inv.flatten()
        # end for idx in range(K)
        print(f"Reconstructed Y_pred_total shape after {imf_col}: {Y_pred_total.shape}")
        print(f"Reconstructed Y_real_total shape after {imf_col}: {Y_real_total.shape}")

        result = (models, Y_real_total, Y_pred_total, stats_t)

    return result

def reconstruct_model(SIGNALS, SIGNAL_NAMES, SEQ_LEN, 
                    PRED_LEN, BATCH_SIZE, TARGET, 
                    DEVICE, CKPT_DIR, u_all, K, model):
    y_preds_inv_ref = None
    y_true_inv_ref = None
    
    for idx in range(K):
        imf_col = f"mode{idx}"

        # build dataframe for this IMF
        modes_df = pd.DataFrame({
            f"{SIGNAL_NAMES[sig_idx]}_{imf_col}": u_all[sig_idx][idx, :]
            for sig_idx in range(len(SIGNALS))
        })

        _, _, test_dl, y_scaler, _ = build_loaders_for_imf(
            df=modes_df, imf_col=imf_col,
            seq_len=SEQ_LEN, pred_len=PRED_LEN, 
            batch=BATCH_SIZE, target_col=TARGET
        )

        # load model
        model_i = model
        ckpt_path = os.path.join(CKPT_DIR, f"model_imf_{idx}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found: {ckpt_path}")
        model_i.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model_i.eval()

        # prediction
        y_pred, y_true = predict_loader(model_i, test_dl, DEVICE)
        y_pred_inv = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_true_inv = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()

        # Reconstruct sum
        if y_preds_inv_ref is None:
            y_preds_inv_ref = y_pred_inv.copy()
            y_true_inv_ref = y_true_inv.copy()
        else:
            n = min(len(y_preds_inv_ref), len(y_pred_inv))
            y_preds_inv_ref[:n] += y_pred_inv[:n]
            y_true_inv_ref[:n] += y_true_inv[:n]

    # Check for None before accessing .shape to avoid AttributeError
    if y_true_inv_ref is None or y_preds_inv_ref is None:
        print("Warning: y_true_inv_ref or y_preds_inv_ref is None. Cannot print shapes.")
    else:
        print(f"shapes of true and predicted values: {y_true_inv_ref.shape}, {y_preds_inv_ref.shape}")

    return y_true_inv_ref, y_preds_inv_ref

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
    __, __, stats_t, vt, vp = training_amp(
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
def print_metrics(Y_real_total, Y_pred_total, excel_file_path):
    abs_errors = np.abs(Y_real_total - Y_pred_total)
    squared_errors = (Y_real_total - Y_pred_total)**2
    df_results = pd.DataFrame({'True_Values': Y_real_total,
                                'Predicted_Values': Y_pred_total,
                                'Absolute_Error': abs_errors,
                                'Squared_Error': squared_errors})
    df_results.to_excel(excel_file_path, index=False)

    # Compute metrics (RMSE) after all IMFs
    R2, MAE, RMSE = compute_metrics(Y_real_total, Y_pred_total)
    tqdm.write(f"\n=== VMD Reconstruction Results ===")
    tqdm.write(f"RMSE: {RMSE:.4f}")
    tqdm.write(f"MAE:  {MAE:.4f}")
    tqdm.write(f"R2:   {R2:.4f}")    