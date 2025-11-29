##################################
#### Imports y setup #############
##################################

import os, sys, warnings, yaml

warnings.filterwarnings("ignore")

# Make imports work whether this file is run as a script or as a module.
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))        # .../src
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))  # repository root
for _p in (SRC_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os
import yaml

import numpy as np
import pandas as pd
from types import SimpleNamespace
from time import perf_counter
from typing import Any, cast

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

# ==== local imports ====
from src.utils.helpers import build_loaders, build_loaders_for_imf
from src.visualization.plots import visualize, scatter
from src.pipeline.metrics import compute_metrics

from models.TimesNet_BiLSTM import TimesNet_BiLSTM_Parallel, BiLSTM, TimesNet
from statsmodels.tsa.seasonal import STL
from src.features.vmd import VMD

# =========================================================
# Settings
# =========================================================
CONFIG_PATH = os.path.join("configs", "parameters.yaml")
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

DEVICE = torch.device(CFG["experiment"]["device"] if torch.cuda.is_available() else "cpu")
SEQ_LEN = int(CFG["data"]["sequence_length"])
PRED_LEN = int(CFG["data"]["prediction_length"])
TARGET = CFG["data"]["target"]
CSV_PATH = CFG["data"]["root_path"]

LR          = float(CFG["experiment"]["learning_rate"])
BATCH_SIZE  = int(CFG["experiment"]["batch_size"])

STL_CFG = CFG["stl"]
VMD_CFG = CFG["vmd"]

print(f"STL enabled: {STL_CFG.get('enabled', True)}")
print(f"VMD enabled: {VMD_CFG.get('enabled', True)}")

# ---- Output directories
CKPT_DIR = os.path.join(CFG["training"]["checkpoint_dir"], 
                        CFG["training"]["model_dir"])
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"==> Config loaded from {CONFIG_PATH}")
print(f"Device: {DEVICE}")
print(f"Checkpoints: {CKPT_DIR}")


# =========================================================
# BUILD MODEL CFG
# =========================================================
def build_model_cfg(cfg_yaml: dict) -> SimpleNamespace:
    m  = cfg_yaml["model"]
    tn = m["timesnet"]
    bl = m["bilstm"]
    return SimpleNamespace(
        seq_len=CFG["data"]["sequence_length"],
        pred_len=CFG["data"]["prediction_length"],
        top_k=tn["top_k"],
        d_model=tn["d_model"],
        d_ff=tn["d_ff"],
        num_kernels=tn["num_kernels"],
        dropout=tn["dropout"],
        num_times_blocks=tn["num_times_blocks"],
        n_features=m["input_features"],
        n_targets=m["output_features"],
        hidden=bl["hidden_size"],
        layers=bl["layers"],
        bidirectional=bl["bidirectional"],
    )

MODEL_CFG = build_model_cfg(CFG)


##################################
# 1) Load dataset
##################################

df = pd.read_csv(CSV_PATH)
assert TARGET in df.columns, f"TARGET '{TARGET}' not found in {CSV_PATH}"

##################################
# 2) STL (optional, according to config)
##################################

if STL_CFG.get("enabled", True):
    # Note: by default calculates STL on the entire series (without rolling).
    # If you want to avoid strict leakage, implement rolling/block in helpers.
    stl = STL(df[TARGET], period=int(STL_CFG["period"]))
    res = stl.fit()
    df["Active_Power_Trend"]    = res.trend
    df["Active_Power_Seasonal"] = res.seasonal
    df["Active_Power_Residual"] = res.resid


##################################
# 3) Build base signals
##################################

signal_0  = df['Wind_Speed']
signal_1  = df['Weather_Temperature_Celsius']
signal_2  = df['Global_Horizontal_Radiation']
signal_3  = df['Max_Wind_Speed']
signal_4  = df['Pyranometer_1']
signal_5  = df['Temperature_Probe_1']
signal_6  = df['Temperature_Probe_2']
signal_7  = df['Active_Energy_Received']

if STL_CFG.get("enabled", True):
    signal_8  = df['Active_Power_Trend']
    signal_9  = df['Active_Power_Seasonal']
    signal_10 = df['Active_Power_Residual']
    signal_11 = df['Active_Power']

    SIGNALS = [
        signal_0, signal_1, signal_2, signal_3, signal_4, signal_5,
        signal_6, signal_7, signal_8, signal_9, signal_10, signal_11
    ]
    SIGNAL_NAMES = [
        'Wind_Speed', 'Weather_Temperature_Celsius', 'Global_Horizontal_Radiation',
        'Max_Wind_Speed', 'Pyranometer_1', 'Temperature_Probe_1', 'Temperature_Probe_2',
        'Active_Energy_Received', 'Active_Power_Trend', 'Active_Power_Seasonal',
        'Active_Power_Residual', 'Active_Power'
    ]
else:
    signal_8  = df['Active_Power']

    SIGNALS = [
        signal_0, signal_1, signal_2, signal_3, signal_4, signal_5,
        signal_6, signal_7, signal_8
    ]
    SIGNAL_NAMES = [
        'Wind_Speed', 'Weather_Temperature_Celsius', 'Global_Horizontal_Radiation',
        'Max_Wind_Speed', 'Pyranometer_1', 'Temperature_Probe_1', 'Temperature_Probe_2',
        'Active_Energy_Received', 'Active_Power'
    ]

# =========================================================
# 3. VMD decomposition (optional, according to config)
# =========================================================

if VMD_CFG.get("enabled", True):
    K = int(VMD_CFG["k"])
    alpha = float(VMD_CFG["alpha"])
    tau   = float(VMD_CFG["tau"])
    DC    = bool(VMD_CFG["dc"])
    init  = int(VMD_CFG["init"])
    tol   = float(VMD_CFG["tol"])
    lambda_param = float(VMD_CFG.get("lambda_param", 0))

    u_all = []  # list of arrays (K, N) per signal

    for sig_idx, signal in enumerate(SIGNALS):
        print(f"Processing signal: {SIGNAL_NAMES[sig_idx]}")
        u_signal, _, _ = VMD(signal, alpha, tau, K, DC, init, tol, lambda_param)
        # u_signal typically comes as (N, K) or similar → normalize to (K, N)
        u_signal = np.array(u_signal).T if np.array(u_signal).shape[0] != K else np.array(u_signal)
        u_all.append(u_signal)
        print(f"Completed VMD for signal: {SIGNAL_NAMES[sig_idx]}")

    print("VMD processing completed for all signals.")
    print(f"Total signals processed: {len(SIGNALS)}")
    print(f"Shape example u_all[0]: {np.array(u_all[0]).shape}")  # (K, N) = (3, _)

    print("VMD complete. Example:", u_all[0].shape)

##################################
# 4) Testing
##################################

loss_fn = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

# prediction
def predict_loader(model, dl, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            with autocast(enabled=(device.type == "cuda")):
                out = model(xb)
            preds.append(out.detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

if VMD_CFG.get("enabled", True):
    y_preds_inv_ref = None
    y_true_inv_ref = None
    per_imf_metrics = []

    t0 = perf_counter()

    for idx in range(K):
        imf_col = f"mode{idx}"

        # build dataframe for this IMF
        modes_df = pd.DataFrame({
            f"{SIGNAL_NAMES[sig_idx]}_{imf_col}": u_all[sig_idx][idx, :]
            for sig_idx in range(len(SIGNALS))
        })

        _, _, test_dl, y_scaler, _ = build_loaders_for_imf(
            df=modes_df, imf_col=imf_col,
            seq_len=SEQ_LEN, pred_len=PRED_LEN, batch=int(CFG["experiment"]["batch_size"])
        )

        # load model
        # model_i = BiLSTM(configs=cast(Any, MODEL_CFG)).to(DEVICE)
        # model_i = TimesNet(configs=cast(Any, MODEL_CFG)).to(DEVICE)
        model_i = TimesNet_BiLSTM_Parallel(configs=cast(Any, MODEL_CFG)).to(DEVICE)
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
            y_true_inv_ref = y_true_inv_ref[:n]

    print(f"shapes of true and predicted values: {y_true_inv_ref.shape}, {y_preds_inv_ref.shape}")

    abs_errors = np.abs(y_true_inv_ref - y_preds_inv_ref)
    squared_errors = (y_true_inv_ref - y_preds_inv_ref)**2
    excel_file_path = "/home/brakine/VMD_STL_Parallel_TimesNet_BiLSTM_POWERFORECASTING/outputs/logs/STL_VMD_TimesNet_BiLSTM/Predictions_STL_VMD_TimesNet_BiLSTM_test.xlsx"
    df_results = pd.DataFrame({'True_Values': y_true_inv_ref,
                                'Predicted_Values': y_preds_inv_ref,
                                'Absolute_Error': abs_errors,
                                'Squared_Error': squared_errors})
    df_results.to_excel(excel_file_path, index=False)


    # =========================================================
    # 5. METRICS 
    # =========================================================
    final_R2, final_MAE, final_RMSE = compute_metrics(y_true_inv_ref, y_preds_inv_ref)

    print("\n=== MMetrics per IMF ===")
    for m in per_imf_metrics:
        print(f"IMF_{m['IMF']}: MAE={m['MAE']:.4f} | RMSE={m['RMSE']:.4f} | R2={m['R2']:.4f} | n={m['n']}")

    print("\n=== FINAL METRICS (Sum of IMFs, real scale) ===")
    print(f"MAE : {final_MAE:.4f}")
    print(f"RMSE: {final_RMSE:.4f}")
    print(f"R²  : {final_R2:.4f}")

    # Plot total prediction (saving optional)
    try:
        visualize(days=1, tt_inv=y_true_inv_ref, tp_inv=y_preds_inv_ref, TARGET="Reconstructed_Sum")
        # plt_recon = os.path.join(PLOTS_DIR, "Reconstructed_Sum_series.png")
        scatter(y_true_inv_ref, y_preds_inv_ref)
        # plt_recon_scatter = os.path.join(PLOTS_DIR, "Reconstructed_Sum_scatter.png")
        # print(f"Final plots saved: {plt_recon}, {plt_recon_scatter}")
    except Exception as e:
        print(f"Warning: final plots could not be generated: {e}")

    print(f"\nProcess completed in {perf_counter() - t0:.2f}s")
else:
    # Load sequences
    train_dl, val_dl, test_dl, y_scaler, n_val_seq = build_loaders(
        df=df, seq_len=SEQ_LEN, pred_len=PRED_LEN, batch=BATCH_SIZE
    )

    # Model + optimizer
    model = TimesNet_BiLSTM_Parallel(configs=cast(Any, MODEL_CFG)).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    model_path = os.path.join(CKPT_DIR, f"STL_TimesNet_BiLSTM_best_model.pt")
    model.load_state_dict(torch.load(model_path, 
                                    map_location=DEVICE))
    model.eval()

    y_pred, y_true = predict_loader(model, test_dl, DEVICE)
    y_pred_inv = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    y_true_inv = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    print(f"shapes of true and predicted values: {y_true_inv.shape}, {y_pred_inv.shape}")

    abs_errors = np.abs(y_true_inv - y_pred_inv)
    squared_errors = (y_true_inv - y_pred_inv)**2
    excel_file_path = "/home/brakine/VMD_STL_Parallel_TimesNet_BiLSTM_POWERFORECASTING/outputs/logs/STL_TimesNet_BiLSTM/Predictions_STL_TimesNet_BiLSTM_test.xlsx"
    df_results = pd.DataFrame({'True_Values': y_true_inv,
                                'Predicted_Values': y_pred_inv,
                                'Absolute_Error': abs_errors,
                                'Squared_Error': squared_errors})
    df_results.to_excel(excel_file_path, index=False)

    # =================================================
    # 5. METRICS    
    # =================================================
    # Compute metrics on the denormalized (original-scale) values
    final_R2, final_MAE, final_RMSE = compute_metrics(y_true=y_true_inv, y_pred=y_pred_inv)

    print("\n=== FINAL METRICS ===")
    print(f"MAE : {final_MAE:.6f}")
    print(f"RMSE: {final_RMSE:.6f}")
    print(f"R²  : {final_R2:.6f}")

    # Plot total prediction (saving optional)
    try:
        visualize(days=1, tt_inv=y_true_inv, tp_inv=y_pred_inv, TARGET="Active Power (KW)")
        # plt_recon = os.path.join(PLOTS_DIR, "Reconstructed_Sum_series.png")
        scatter(y_true_inv, y_pred_inv)
        # plt_recon_scatter = os.path.join(PLOTS_DIR, "Reconstructed_Sum_scatter.png")
        # print(f"Final plots saved: {plt_recon}, {plt_recon_scatter}")
    except Exception as e:
        print(f"Warning: final plots could not be generated: {e}")