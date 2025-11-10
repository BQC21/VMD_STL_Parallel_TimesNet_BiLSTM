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
from src.utils.helpers import build_loaders_for_imf
from src.visualization.plots import visualize, scatter
from src.pipeline.metrics import compute_metrics

from src.models.TimesNet_BiLSTM import TimesNet_BiLSTM_Parallel
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

STL_CFG = CFG["stl"]
VMD_CFG = CFG["vmd"]
K = int(VMD_CFG["k"])

CKPT_DIR = os.path.join(
    CFG["training"]["checkpoint_dir"],
    f'VMD_TimesNet_BiLSTM/k{K}_alpha{VMD_CFG["alpha"]}_lambda{VMD_CFG["lambda_param"]}'
)
PLOTS_DIR = os.path.join("outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

print(f"==> Config loaded from {CONFIG_PATH}")
print(f"Device: {DEVICE}")
print(f"Checkpoints: {CKPT_DIR}")
print(f"Plots: {PLOTS_DIR}")


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


# =========================================================
# 1. Load dataset
# =========================================================
df = pd.read_csv(CSV_PATH)
assert TARGET in df.columns, f"TARGET '{TARGET}' not found in {CSV_PATH}"
print(f"Dataset loaded: {df.shape}")

if STL_CFG.get("enabled", True):
    stl = STL(df[TARGET], period=int(STL_CFG["period"]), robust=bool(STL_CFG["robust"]))
    res = stl.fit()
    df["Active_Power_Trend"]    = res.trend
    df["Active_Power_Seasonal"] = res.seasonal
    df["Active_Power_Residual"] = res.resid
else:
    df["Active_Power_Trend"]    = 0.0
    df["Active_Power_Seasonal"] = 0.0
    df["Active_Power_Residual"] = df[TARGET].values

# =========================================================
# 2. Base signals
# =========================================================
SIGNALS = [
    df['Wind_Speed'],
    df['Weather_Temperature_Celsius'],
    df['Global_Horizontal_Radiation'],
    df['Max_Wind_Speed'],
    df['Pyranometer_1'],
    df['Temperature_Probe_1'],
    df['Temperature_Probe_2'],
    df['Active_Energy_Received'],
    df['Active_Power_Trend'],
    df['Active_Power_Seasonal'],
    df['Active_Power_Residual'],
    df['Active_Power'],
]
SIGNAL_NAMES = [
    'Wind_Speed', 'Weather_Temperature_Celsius', 'Global_Horizontal_Radiation',
    'Max_Wind_Speed', 'Pyranometer_1', 'Temperature_Probe_1', 'Temperature_Probe_2',
    'Active_Energy_Received', 'Active_Power_Trend', 'Active_Power_Seasonal',
    'Active_Power_Residual', 'Active_Power'
]

# =========================================================
# 3. VMD decomposition
# =========================================================

alpha = float(VMD_CFG["alpha"])
tau   = float(VMD_CFG["tau"])
DC    = bool(VMD_CFG["dc"])
init  = int(VMD_CFG["init"])
tol   = float(VMD_CFG["tol"])
lambda_param = float(VMD_CFG.get("lambda_param", 0))

u_all = []
for sig_idx, signal in enumerate(SIGNALS):
    print(f"VMD → {SIGNAL_NAMES[sig_idx]}")
    u_signal, _, _ = VMD(signal, alpha, tau, K, DC, init, tol, lambda_param)
    u_signal = np.array(u_signal)
    if u_signal.shape[0] != K and u_signal.shape[1] == K:
        u_signal = u_signal.T
    assert u_signal.shape[0] == K, f"Unexpected IMF shape for {SIGNAL_NAMES[sig_idx]}"
    u_all.append(u_signal)

print("VMD complete. Example:", u_all[0].shape)


# =========================================================
# 4. PREDICTION PER IMF
# =========================================================
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


sum_preds_inv = None
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

    # Individual IMF metrics
    R2, MAE, RMSE = compute_metrics(y_true_inv, y_pred_inv)
    per_imf_metrics.append({
        "IMF": idx,
        "MAE": MAE,
        "RMSE": RMSE,
        "R2": R2,
        "n": len(y_true_inv)
    })

    # # plots for this IMF (optional)
    # try:
    #     visualize(days=3, tt_inv=y_true_inv, tp_inv=y_pred_inv, TARGET=f"IMF_{idx}")
    #     plt_path = os.path.join(PLOTS_DIR, f"IMF_{idx}_series.png")
    #     scatter(y_true_inv, y_pred_inv)
    #     plt_scatter = os.path.join(PLOTS_DIR, f"IMF_{idx}_scatter.png")
    #     print(f"Saved plots: {plt_path}, {plt_scatter}")
    # except Exception as e:
    #     print(f"warning: Plots could not be generated for IMF_{idx}: {e}")

    # Reconstruct sum
    if sum_preds_inv is None:
        sum_preds_inv = y_pred_inv.copy()
        y_true_inv_ref = y_true_inv.copy()
    else:
        n = min(len(sum_preds_inv), len(y_pred_inv))
        sum_preds_inv[:n] += y_pred_inv[:n]
        y_true_inv_ref = y_true_inv_ref[:n]

# =========================================================
# 5. METRICS 
# =========================================================
final_R2, final_MAE, final_RMSE = compute_metrics(y_true_inv_ref, sum_preds_inv)

print("\n=== MMetrics per IMF ===")
for m in per_imf_metrics:
    print(f"IMF_{m['IMF']}: MAE={m['MAE']:.4f} | RMSE={m['RMSE']:.4f} | R2={m['R2']:.4f} | n={m['n']}")

print("\n=== FINAL METRICS (Sum of IMFs, real scale) ===")
print(f"MAE : {final_MAE:.6f}")
print(f"RMSE: {final_RMSE:.6f}")
print(f"R²  : {final_R2:.6f}")

# Plot total prediction (saving optional)
try:
    visualize(days=4, tt_inv=y_true_inv_ref, tp_inv=sum_preds_inv, TARGET="Reconstructed_Sum")
    # plt_recon = os.path.join(PLOTS_DIR, "Reconstructed_Sum_series.png")
    scatter(y_true_inv_ref, sum_preds_inv)
    # plt_recon_scatter = os.path.join(PLOTS_DIR, "Reconstructed_Sum_scatter.png")
    # print(f"Final plots saved: {plt_recon}, {plt_recon_scatter}")
except Exception as e:
    print(f"Warning: final plots could not be generated: {e}")

print(f"\nProcess completed in {perf_counter() - t0:.2f}s")