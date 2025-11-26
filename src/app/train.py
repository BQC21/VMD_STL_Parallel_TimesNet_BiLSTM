##################################
#### Frameworks and setup ########
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

# Utils 
from utils.helpers import build_loaders, training_amp
from pipeline.metrics import compute_metrics

# DL model
from models.TimesNet_BiLSTM import TimesNet_BiLSTM_Parallel

# STL
from statsmodels.tsa.seasonal import STL

# Computational time
from time import perf_counter
from types import SimpleNamespace
from typing import Any, cast
from tqdm import tqdm

##################################
# Load YAML configuration
##################################

CONFIG_PATH = os.path.join("configs", "parameters.yaml")
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

# ---- seeds 
SEED = int(CFG["experiment"]["seed"])
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---- device
DEVICE = torch.device(CFG["experiment"]["device"] if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__)
print("torch.version.cuda:", getattr(getattr(torch, 'version', None), 'cuda', None))
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Device:", DEVICE)

# ---- basic hyperparameters
EPOCHS      = int(CFG["experiment"]["epochs"])
BATCH_SIZE  = int(CFG["experiment"]["batch_size"])
LR          = float(CFG["experiment"]["learning_rate"])
PATIENCE_ES = int(CFG["experiment"]["early_stopping_patience"])

SEQ_LEN = int(CFG["data"]["sequence_length"])
PRED_LEN = int(CFG["data"]["prediction_length"])
TARGET = CFG["data"]["target"]
CSV_PATH = CFG["data"]["root_path"]

# ---- STL config 
STL_CFG = CFG["stl"]


# ---- Output directories
CKPT_DIR = os.path.join(CFG["training"]["checkpoint_dir"], 
                        f'TimesNet_BiLSTM/period{STL_CFG["period"]}/')
os.makedirs(CKPT_DIR, exist_ok=True)


##################################
# Helper: convert YAML to object expected by the model
##################################

def build_model_cfg(cfg_yaml: dict) -> SimpleNamespace:
    m  = cfg_yaml["model"]
    tn = m["timesnet"]
    bl = m["bilstm"]
    # namespaces
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
    signal_8 = df['Active_Power']

    SIGNALS = [
        signal_0, signal_1, signal_2, signal_3, signal_4, signal_5,
        signal_6, signal_7, signal_8
    ]
    SIGNAL_NAMES = [
        'Wind_Speed', 'Weather_Temperature_Celsius', 'Global_Horizontal_Radiation',
        'Max_Wind_Speed', 'Pyranometer_1', 'Temperature_Probe_1', 'Temperature_Probe_2',
        'Active_Energy_Received', 'Active_Power'
    ]


##################################
# 4) Training
##################################

loss_fn = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

tqdm.write(f"\n=== Training ===")
t0 = perf_counter()

Y_pred_total = np.zeros(int(0.2*len(df)-SEQ_LEN-PRED_LEN+1)) 
Y_real_total = np.zeros(int(0.2*len(df)-SEQ_LEN-PRED_LEN+1))
print(f"Initial Y_total shape: {Y_pred_total.shape}")
print(f"Initial Y_real_total shape: {Y_real_total.shape}")

# Load sequences
train_dl, val_dl, test_dl, __, __ = build_loaders(
    df=df, seq_len=SEQ_LEN, pred_len=PRED_LEN, batch=BATCH_SIZE
)

# Model + optimizer
model = TimesNet_BiLSTM_Parallel(configs=cast(Any, MODEL_CFG)).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
model_path = os.path.join(CKPT_DIR, f"TimesNet_BiLSTM_best_model.pt")

# Training with AMP + early stopping
__, __, stats_t, vt, vp = training_amp(
    model=model, device=str(DEVICE), 
    loss_fn=loss_fn, scaler=scaler,
    optim=optim, train_dl=train_dl, val_dl=val_dl,
    MODEL_PATH=model_path, df = df, 
    seq_len=SEQ_LEN, pred_len=PRED_LEN,
    patience=PATIENCE_ES, verbose=True
)

print(f"shapes of true and predicted values: {vt.shape}, {vp.shape}")
# Computer metrics (RMSE)
__, __, RMSE = compute_metrics(vt, vp)
tqdm.write(f"RMSE: {RMSE:.4f}")

# Summary times 
t_final = perf_counter() - t0
tqdm.write(f"[Total time ={t_final:.2f}s | epoch_avg={stats_t.get('epoch_avg_s', np.nan):.2f}s | val_avg={stats_t.get('val_avg_s', np.nan):.2f}s")