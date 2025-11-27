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
from src.utils.helpers import build_loaders
from src.visualization.plots import visualize, scatter
from src.pipeline.metrics import compute_metrics

from models.TimesNet_BiLSTM import TimesNet_BiLSTM_Parallel, BiLSTM, TimesNet
from statsmodels.tsa.seasonal import STL

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
    stl = STL(df[TARGET], period=int(STL_CFG["period"]), robust=bool(STL_CFG["robust"]))
    res = stl.fit()
    df["Active_Power_Trend"]    = res.trend
    df["Active_Power_Seasonal"] = res.seasonal
    df["Active_Power_Residual"] = res.resid


##################################
# 3) Build base signals (12 in total)
##################################

signal_0  = df['Total solar irradiance (W/m2)']
signal_1  = df['Air temperature  (°C) ']
signal_2  = df['Relative humidity (%)']

if STL_CFG.get("enabled", True):
    signal_3  = df['Active_Power_Trend']
    signal_4  = df['Active_Power_Seasonal']
    signal_5 = df['Active_Power_Residual']
    signal_6 = df[TARGET]

    SIGNALS = [
        signal_0, signal_1, signal_2, signal_3, signal_4, signal_5, signal_6
    ]
    SIGNAL_NAMES = [
        'Total solar irradiance (W/m2)', 'Air temperature  (°C) ', 'Relative humidity (%)',
        'Active_Power_Trend', 'Active_Power_Seasonal','Active_Power_Residual', 'Power (MW)'
    ]
else:
    signal_3 = df['Power (MW)']

    SIGNALS = [
        signal_0, signal_1, signal_2, signal_3
    ]
    SIGNAL_NAMES = [
        'Total solar irradiance (W/m2)', 'Air temperature (°C)', 
        'Relative humidity (%)', 'Power (MW)'
    ]



##################################
# 4) Testing
##################################

loss_fn = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))


# Load sequences
train_dl, val_dl, test_dl, y_scaler, n_val_seq = build_loaders(
    df=df, seq_len=SEQ_LEN, pred_len=PRED_LEN, batch=BATCH_SIZE
)

# Model + optimizer
# model = BiLSTM(configs=cast(Any, MODEL_CFG)).to(DEVICE)
model = TimesNet(configs=cast(Any, MODEL_CFG)).to(DEVICE)
# model = TimesNet_BiLSTM_Parallel(configs=cast(Any, MODEL_CFG)).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

model_path = os.path.join(CKPT_DIR, f"TimesNet_best_model.pt")
model.load_state_dict(torch.load(model_path, 
                                map_location=DEVICE))
model.eval()

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


y_pred, y_true = predict_loader(model, test_dl, DEVICE)
y_pred_inv = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
y_true_inv = y_scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
print(f"shapes of true and predicted values: {y_true_inv.shape}, {y_pred_inv.shape}")

abs_errors = np.abs(y_true_inv - y_pred_inv)
squared_errors = (y_true_inv - y_pred_inv)**2
excel_file_path = "/home/brakine/VMD_STL_Parallel_TimesNet_BiLSTM_POWERFORECASTING/outputs/logs/TimesNet/Predictions_TimesNet_test.xlsx"
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
    visualize(days=1, tt_inv=y_true_inv, tp_inv=y_pred_inv, TARGET="Power (MW)")
    # plt_recon = os.path.join(PLOTS_DIR, "Reconstructed_Sum_series.png")
    scatter(y_true_inv, y_pred_inv)
    # plt_recon_scatter = os.path.join(PLOTS_DIR, "Reconstructed_Sum_scatter.png")
    # print(f"Final plots saved: {plt_recon}, {plt_recon_scatter}")
except Exception as e:
    print(f"Warning: final plots could not be generated: {e}")
