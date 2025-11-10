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
from utils.helpers import build_loaders_for_imf, training_amp

# DL model
from models.TimesNet_BiLSTM import TimesNet_BiLSTM_Parallel

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

# ---- STL / VMD config
STL_CFG = CFG["stl"]
VMD_CFG = CFG["vmd"]
K = int(VMD_CFG["k"])


# ---- Output directories
CKPT_DIR = os.path.join(CFG["training"]["checkpoint_dir"], f'VMD_TimesNet_BiLSTM/k{K}_alpha{VMD_CFG["alpha"]}_lambda{VMD_CFG["lambda_param"]}')
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
else:
    df["Active_Power_Trend"]    = 0.0
    df["Active_Power_Seasonal"] = 0.0
    df["Active_Power_Residual"] = df[TARGET].values


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

##################################
# 4) VMD per signal
##################################

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
print(f"Shape example u_all[0]: {np.array(u_all[0]).shape}")  # (K, N)

##################################
# 5) Training per IMF
##################################

loss_fn = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

models = []
tiempos_imf = []

for idx in range(K):
    imf_col = f"mode{idx}"

    # build dataframe for this IMF
    modes_df = pd.DataFrame({
        f"{SIGNAL_NAMES[sig_idx]}_{imf_col}": u_all[sig_idx][idx, :]
        for sig_idx in range(len(SIGNALS))
    })

    tqdm.write(f"\n=== Training {imf_col} ===")
    t0_imf = perf_counter()

    # Load sequences
    train_dl, val_dl, test_dl, y_scaler, n_val_seq = build_loaders_for_imf(
        df=modes_df, imf_col=imf_col,
        seq_len=SEQ_LEN, pred_len=PRED_LEN, batch=BATCH_SIZE
    )

    # Model + optimizer
    model_i = TimesNet_BiLSTM_Parallel(configs=cast(Any, MODEL_CFG)).to(DEVICE)
    optim_i = torch.optim.Adam(model_i.parameters(), lr=LR, weight_decay=1e-4)
    model_path = os.path.join(CKPT_DIR, f"model_imf_{idx}.pt")

    # Training with AMP + early stopping
    _lt, _lv, stats_t = training_amp(
        model=model_i, device=str(DEVICE), loss_fn=loss_fn, scaler=scaler,
        optim=optim_i, train_dl=train_dl, val_dl=val_dl,
        MODEL_PATH=model_path, epochs=EPOCHS, patience=PATIENCE_ES, verbose=True
    )

    # Plot losses (optional)
    # try:
    #     plot_losses(_lt, _lv, imf_col=imf_col)
    # except Exception as e:
    #     print(f"Warning: could not plot losses for {imf_col}: {e}")
    models.append(model_i)

    t_imf = perf_counter() - t0_imf
    tiempos_imf.append(t_imf)
    tqdm.write(f"[{imf_col}] Total time ={t_imf:.2f}s | epoch_avg={stats_t.get('epoch_avg_s', np.nan):.2f}s | val_avg={stats_t.get('val_avg_s', np.nan):.2f}s")

# Summary times per IMF
tqdm.write("\n=== Summary times per IMF ===")
for i, t in enumerate(tiempos_imf):
    tqdm.write(f"IMF_{i}: {t:.2f}s")
tqdm.write(f"Total time (all IMFs): {sum(tiempos_imf):.2f}s")