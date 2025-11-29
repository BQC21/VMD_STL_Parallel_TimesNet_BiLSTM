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
from utils.helpers import build_loaders, training_amp, build_loaders_for_imf
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
    if CSV_PATH.endswith("figshare_modified.csv"):
        # Note: by default calculates STL on the entire series (without rolling).
        # If you want to avoid strict leakage, implement rolling/block in helpers.
        stl = STL(df[TARGET], period=int(STL_CFG["period"]), robust=bool(STL_CFG["robust"]))
        res = stl.fit()
        df["Power_MW_Trend"]    = res.trend
        df["Power_MW_Seasonal"] = res.seasonal
        df["Power_MW_Residual"] = res.resid
        print(f"STL decomposition completed \n")
    else: # DKASC
        # Note: by default calculates STL on the entire series (without rolling).
        # If you want to avoid strict leakage, implement rolling/block in helpers.
        stl = STL(df[TARGET], period=int(STL_CFG["period"]))
        res = stl.fit()
        df["Active_Power_Trend"]    = res.trend
        df["Active_Power_Seasonal"] = res.seasonal
        df["Active_Power_Residual"] = res.resid
        print(f"STL decomposition completed \n")


##################################
# 3) Build base signals 
##################################

if CSV_PATH.endswith("figshare_modified.csv"):
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
        signal_3 = df[TARGET]

        SIGNALS = [
            signal_0, signal_1, signal_2, signal_3
        ]
        SIGNAL_NAMES = [
            'Total solar irradiance (W/m2)', 'Air temperature (°C)', 
            'Relative humidity (%)', 'Power (MW)'
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

loss_fn = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

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

if VMD_CFG.get("enabled", True):
    models = []
    tiempos_imf = []

Y_pred_total = np.zeros(int(0.2*len(df)-SEQ_LEN-PRED_LEN+1))  # Adjust size as needed (20% of data length)
Y_real_total = np.zeros(int(0.2*len(df)-SEQ_LEN-PRED_LEN+1)) 
print(f"Initial Y_total shape: {Y_pred_total.shape}")
print(f"Initial Y_real_total shape: {Y_real_total.shape} \n")

tqdm.write(f"\n=== Training ===")
t0 = perf_counter()

if VMD_CFG.get("enabled", True):
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

        # --------- Model + optimizer
        model_i = TimesNet_BiLSTM_Parallel(configs=cast(Any, MODEL_CFG)).to(DEVICE)
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

    abs_errors = np.abs(Y_real_total - Y_pred_total)
    squared_errors = (Y_real_total - Y_pred_total)**2
    excel_file_path = "/home/brakine/VMD_STL_Parallel_TimesNet_BiLSTM_POWERFORECASTING/outputs/logs/STL_VMD_TimesNet_BiLSTM/Predictions_STL_VMD_TimesNet_BiLSTM_valid.xlsx"
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

    ##################################
    # 4) Testing
    ##################################
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

    y_pred_test, y_true_test = predict_loader(model, test_dl, DEVICE)
    y_pred_inv_test = y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
    y_true_inv_test = y_scaler.inverse_transform(y_true_test.reshape(-1, 1)).ravel()
    print(f"shapes of true and predicted values: {y_true_inv_test.shape}, {y_pred_inv_test.shape}")

    abs_errors = np.abs(y_true_inv_test - y_pred_inv_test)
    squared_errors = (y_true_inv_test - y_pred_inv_test)**2
    excel_file_path = os.path.join(CFG["training"]["log_dir"], 
                        CFG["training"]["model_dir"],
                        "Predictions_STL_TimesNet_BiLSTM_test.xlsx")
    df_results = pd.DataFrame({'True_Values': y_true_inv_test,
                                'Predicted_Values': y_pred_inv_test,
                                'Absolute_Error': abs_errors,
                                'Squared_Error': squared_errors})
    df_results.to_excel(excel_file_path, index=False)

    # Computer metrics (RMSE)
    R2, MAE, RMSE = compute_metrics(y_true_inv_test, y_pred_inv_test)
    tqdm.write(f"RMSE: {RMSE:.4f}")
    tqdm.write(f"MAE:  {MAE:.4f}")
    tqdm.write(f"R2:   {R2:.4f}")
    tqdm.write(f"\n")

    # Plot total prediction (saving optional)
    try:
        visualize(days=1, tt_inv=y_true_inv_test, tp_inv=y_pred_inv_test, TARGET=TARGET)
        plt_history = os.path.join(CFG["training"]["plot_dir"],
                                CFG["training"]["model_dir"],
                                "Predictions_STL_VMD_TimesNet_BiLSTM_history.png")
        scatter(y_true_inv_test, y_pred_inv_test)
        plt_scatter = os.path.join(CFG["training"]["plot_dir"],
                            CFG["training"]["model_dir"],
                            "Predictions_STL_VMD_TimesNet_BiLSTM_scatter.png")
        print(f"Final plots saved: {plt_history}, {plt_scatter}")
    except Exception as e:
        print(f"Warning: final plots could not be generated: {e}")

    print(f"\nProcess completed in {perf_counter() - t0:.2f}s")

else: # no VMD
    # Load sequences
    train_dl, val_dl, test_dl, y_scaler, __ = build_loaders(
        df=df, seq_len=SEQ_LEN, pred_len=PRED_LEN, 
        batch=BATCH_SIZE, target_col=TARGET
    )

    # -------- Model + optimizer
    model = TimesNet_BiLSTM_Parallel(configs=cast(Any, MODEL_CFG)).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Use a filename that doesn't depend on VMD-specific variables when VMD is disabled
    model_path = os.path.join(CKPT_DIR, "STL_TimesNet_BiLSTM_best_model.pt")

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

    abs_errors = np.abs(y_true_inv - y_pred_inv)
    squared_errors = (y_true_inv - y_pred_inv)**2
    excel_file_path = os.path.join(CFG["training"]["log_dir"], 
                        CFG["training"]["model_dir"],
                        "Predictions_STL_TimesNet_BiLSTM_valid.xlsx")
    df_results = pd.DataFrame({'True_Values': y_true_inv,
                                'Predicted_Values': y_pred_inv,
                                'Absolute_Error': abs_errors,
                                'Squared_Error': squared_errors})
    df_results.to_excel(excel_file_path, index=False)

    print(f"shapes of true and predicted values: {y_true_inv.shape}, {y_pred_inv.shape}\n")
    # Computer metrics (RMSE)
    R2, MAE, RMSE = compute_metrics(y_true_inv, y_pred_inv)
    tqdm.write(f"RMSE: {RMSE:.4f}")
    tqdm.write(f"MAE:  {MAE:.4f}")
    tqdm.write(f"R2:   {R2:.4f}")
    tqdm.write(f"\n")

    # Summary times 
    t_final = perf_counter() - t0
    tqdm.write(f"[Total time ={t_final:.2f}s | epoch_avg={stats_t.get('epoch_avg_s', np.nan):.2f}s | val_avg={stats_t.get('val_avg_s', np.nan):.2f}s")

    ##################################
    # 4) Testing
    ##################################
    model.load_state_dict(torch.load(model_path, 
                                    map_location=DEVICE))
    model.eval()

    y_pred_test, y_true_test = predict_loader(model, test_dl, DEVICE)
    y_pred_inv_test = y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
    y_true_inv_test = y_scaler.inverse_transform(y_true_test.reshape(-1, 1)).ravel()
    print(f"shapes of true and predicted values: {y_true_inv_test.shape}, {y_pred_inv_test.shape}")

    abs_errors = np.abs(y_true_inv_test - y_pred_inv_test)
    squared_errors = (y_true_inv_test - y_pred_inv_test)**2
    excel_file_path = os.path.join(CFG["training"]["log_dir"], 
                        CFG["training"]["model_dir"],
                        "Predictions_STL_TimesNet_BiLSTM_test.xlsx")
    df_results = pd.DataFrame({'True_Values': y_true_inv_test,
                                'Predicted_Values': y_pred_inv_test,
                                'Absolute_Error': abs_errors,
                                'Squared_Error': squared_errors})
    df_results.to_excel(excel_file_path, index=False)

    # Computer metrics (RMSE)
    R2, MAE, RMSE = compute_metrics(y_true_inv_test, y_pred_inv_test)
    tqdm.write(f"RMSE: {RMSE:.4f}")
    tqdm.write(f"MAE:  {MAE:.4f}")
    tqdm.write(f"R2:   {R2:.4f}")
    tqdm.write(f"\n")

    # Plot total prediction (saving optional)
    try:
        visualize(days=1, tt_inv=y_true_inv_test, tp_inv=y_pred_inv_test, TARGET=TARGET)
        plt_history = os.path.join(CFG["training"]["plot_dir"],
                                CFG["training"]["model_dir"],
                                "Predictions_STL_TimesNet_BiLSTM_history.png")
        scatter(y_true_inv_test, y_pred_inv_test)
        plt_scatter = os.path.join(CFG["training"]["plot_dir"],
                            CFG["training"]["model_dir"],
                            "Predictions_STL_TimesNet_BiLSTM_scatter.png")
        print(f"Final plots saved: {plt_history}, {plt_scatter}")
    except Exception as e:
        print(f"Warning: final plots could not be generated: {e}")

    print(f"\nProcess completed in {perf_counter() - t0:.2f}s")