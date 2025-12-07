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
from utils.helpers import predict_loader, train_for_each_imf, print_metrics, reconstruct_model
from visualization.plots import visualize, scatter

# DL model
from models.TimesNet_BiLSTM import TimesNet_BiLSTM_Parallel

# STL
from statsmodels.tsa.seasonal import STL

# VMD
from features.vmd import VMD

# Computational time
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

print(f"==> Config loaded from {CONFIG_PATH} \n")

# ---- device
DEVICE = torch.device(CFG["experiment"]["device"] if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__)
print("torch.version.cuda:", getattr(getattr(torch, 'version', None), 'cuda', None))
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Device: \n", DEVICE)

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
VMD_CFG = CFG["vmd"]

print(f"STL enabled: {STL_CFG.get('enabled', True)}")
print(f"VMD enabled: {VMD_CFG.get('enabled', True)}\n")

# ---- Output directories
CKPT_DIR = os.path.join(CFG["training"]["checkpoint_dir"], 
                        CFG["training"]["model_dir"])
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
print(f"Dataset loaded from {CSV_PATH} \n")

##################################
# 2) STL (optional, according to config)
##################################

if STL_CFG.get("enabled", True):
    if CSV_PATH.endswith("figshare_processed.csv"):
        stl = STL(df[TARGET], period=int(STL_CFG["period"]))
        res = stl.fit()
        df["Power_MW_Trend"]    = res.trend
        df["Power_MW_Seasonal"] = res.seasonal
        df["Power_MW_Residual"] = res.resid
        print(f"STL decomposition completed \n")
    else: # DKASC
        stl = STL(df[TARGET], period=int(STL_CFG["period"]))
        res = stl.fit()
        df["Active_Power_Trend"]    = res.trend
        df["Active_Power_Seasonal"] = res.seasonal
        df["Active_Power_Residual"] = res.resid
        print(f"STL decomposition completed \n")


##################################
# 3) Build base signals 
##################################

if CSV_PATH.endswith("figshare_processed.csv"):
    signal_0  = df['Total solar irradiance (W/m2)']
    signal_1  = df['Air temperature  (°C) ']
    signal_2  = df['Relative humidity (%)']
    signal_3  = df['Power_MW_Trend']
    signal_4  = df['Power_MW_Seasonal']
    signal_5 = df['Power_MW_Residual']
    signal_6 = df[TARGET]

    SIGNALS = [
        signal_0, signal_1, signal_2, signal_3, signal_4, signal_5, signal_6
    ]
    SIGNAL_NAMES = [
        'Total solar irradiance (W/m2)', 'Air temperature  (°C) ', 'Relative humidity (%)',
        'Active_Power_Trend', 'Active_Power_Seasonal','Active_Power_Residual', 'Power (MW)'
    ]
else: # DKASC:
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

print(f"Total signals used: {len(SIGNALS)}")
print(f"Signal names: {SIGNAL_NAMES} \n")

##################################
# 4) VMD per signal (optional, according to config)
##################################

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
    print(f"Shape example u_all[0]: {np.array(u_all[0]).shape} \n")  # (K, N) = (3, _)

##################################
# 5) Training
##################################

model_name = CFG["model"]["name"].lower()

if model_name in ["timesnet_bilstm", "timesnet_bilstm_parallel"]:
    model = TimesNet_BiLSTM_Parallel(configs=cast(Any, MODEL_CFG)).to(DEVICE)
else:
    raise ValueError(f"Model not supported: {CFG['model']['name']}")

loss_fn = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

if CSV_PATH.endswith("figshare_processed.csv"):
    Y_real_total = np.zeros(6893) # Figshare
    Y_pred_total = np.zeros(6893) # Figshare    
else: # DKASC
    Y_real_total = np.zeros(int(0.2*len(df)-SEQ_LEN-PRED_LEN+1))
    Y_pred_total = np.zeros(int(0.2*len(df)-SEQ_LEN-PRED_LEN+1))
    
print(f"Initial Y_total shape: {Y_pred_total.shape}")
print(f"Initial Y_real_total shape: {Y_real_total.shape} \n")

tqdm.write(f"\n=== Training ===")

if VMD_CFG.get("enabled", True):
    #### Training for each IMF ####
    result = train_for_each_imf(df, model, u_all, SIGNALS, 
                                SIGNAL_NAMES, DEVICE, CKPT_DIR, 
                                SEQ_LEN, PRED_LEN, 
                                BATCH_SIZE, TARGET, 
                                K, Y_pred_total, Y_real_total,
                                LR, EPOCHS, PATIENCE_ES, loss_fn, scaler)

    if result is None or not isinstance(result, (list, tuple)) or len(result) < 4:
        raise ValueError("train_for_each_imf did not return expected values (models, Y_real_total, Y_pred_total, ...)")

    models, Y_real_total, Y_pred_total, _ = result

    #### Validation metrics ####

    print_metrics(Y_real_total, Y_pred_total)

    #### Testing #######
    y_true_inv_ref, y_preds_inv_ref = reconstruct_model(SIGNALS, SIGNAL_NAMES, SEQ_LEN, 
                    PRED_LEN, BATCH_SIZE, TARGET, 
                    DEVICE, CKPT_DIR, u_all, K, model)
    
    print_metrics(y_true_inv_ref, y_preds_inv_ref)

    # Plot total prediction (saving optional)
    try:
        visualize(days=1, tt_inv=y_true_inv_ref, tp_inv=y_preds_inv_ref, TARGET=TARGET)
        scatter(y_true_inv_ref, y_preds_inv_ref)
    except Exception as e:
        print(f"Warning: final plots could not be generated: {e}")