# NumPy / Pandas / Matplotlib
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

#################### DKASC HISTOGRAM ############################################

# model names 
model_names = ["BiLSTM", "TimesNet", "TimesNet-BiLSTM", 
                    "STL_BiLSTM", "STL_TimesNet", "STL_TimesNet-BiLSTM",
                    "VMD_BiLSTM", "VMD_TimesNet", "VMD_TimesNet-BiLSTM",
                    "STL_VMD_BiLSTM", "STL_VMD_TimesNet", "STL_VMD_TimesNet-BiLSTM"]

# MAE, RMSE, R2 for DKSAC (from "all_exp" branch)
MAE_dksac = [10.2397, 9.3082, 9.4747, 
5.9110, 5.5962, 5.5633, 
3.5010, 4.6414, 4.0848, 
3.3669, 5.1346, 3.6979]
RMSE_dksac = [19.1686, 18.6095, 18.6478, 
11.8351, 9.3876, 9.2110, 
5.6328, 6.8130, 6.3868, 
6.1103, 7.3156, 6.4245]
R2_dksac = [0.9509, 0.9538, 0.9536, 
0.9813, 0.9886, 0.9887, 
0.9956, 0.9936, 0.9944, 
0.9949, 0.9927, 0.9943]

# DF of logs for DKSAC
DKASC = pd.DataFrame({"Model_name": model_names,
                    "MAE": MAE_dksac,
                    "RMSE": RMSE_dksac,
                    "R2": R2_dksac})

model_names_dksac = DKASC["Model_name"].values
x_dksac = np.arange(len(model_names_dksac))  

metrics_dksac = ["MAE", "RMSE", "R2"]

fig_dksac, axes_dksac = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# keep axes iterable even when Matplotlib returns a single Axes instance
axes_iter_dksac: Iterable[Axes]
if isinstance(axes_dksac, np.ndarray):
    axes_iter_dksac = axes_dksac.ravel().tolist()
else:
    axes_iter_dksac = [axes_dksac]

# make histograms for each metric
for ax, metric in zip(axes_iter_dksac, metrics_dksac):
    ax.bar(x_dksac, DKASC[metric])
    ax.set_title(metric, fontsize=14)
    ax.set_xticks(x_dksac)
    ax.set_xticklabels(model_names_dksac, rotation=90)
    ax.set_xlabel("Models")
    ax.set_ylabel(metric)

fig_dksac.suptitle("Comparison of metrics by model (DKSAC)", fontsize=16)
plt.show()

##################### FIGSHARE HISTOGRAM ###########################################

#MAE, RMSE, R2 for figshare (from "all_exp" branch)
MAE_figshare = [3.1638, 3.0838, 3.0284, 
0.5038, 0.4232, 0.4110, 
3.0713, 3.0395, 2.9531, 
0.2955, 0.3400, 0.3154]
RMSE_figshare = [4.4448, 4.3843, 4.2961, 
0.6948, 0.5554, 0.5472, 
4.5132, 4.3344, 4.2022, 
0.3973, 0.4451, 0.4174]
R2_figshare = [0.6079, 0.6184, 0.6336, 
0.9904, 0.9939, 0.9941, 
0.5900, 0.6218, 0.6445, 
0.9968, 0.9960, 0.9965]

# DF of logs for figshare
figshare = pd.DataFrame({"Model_name": model_names,
                    "MAE": MAE_figshare,
                    "RMSE": RMSE_figshare,
                    "R2": R2_figshare})

model_names_figshare = figshare["Model_name"].values
x_figshare = np.arange(len(model_names_figshare))  

metrics_figshare = ["MAE", "RMSE", "R2"]


fig_figshare, axes_figshare = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

# keep axes iterable even when Matplotlib returns a single Axes instance
axes_iter_figshare: Iterable[Axes]
if isinstance(axes_figshare, np.ndarray):
    axes_iter_figshare = axes_figshare.ravel().tolist()
else:
    axes_iter_figshare = [axes_figshare]

# make histograms for each metric
for ax, metric in zip(axes_iter_figshare, metrics_figshare):
    ax.bar(x_figshare, figshare[metric])    
    ax.set_title(metric, fontsize=14)
    ax.set_xticks(x_figshare)
    ax.set_xticklabels(model_names_figshare, rotation=90)
    ax.set_xlabel("Models")
    ax.set_ylabel(metric)

fig_figshare.suptitle("Comparison of metrics by model (figshare)", fontsize=16)
plt.show()