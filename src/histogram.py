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
MAE_dksac = [10.0651, 9.8296, 10.1088, 
6.0830, 5.3157, 5.5533, 
4.4302, 5.5409, 5.7206, 
4.2659, 3.9706, 3.9234]
RMSE_dksac = [19.1476, 18.6770, 18.9476, 
11.8616, 8.9938, 9.3104, 
7.5547, 10.0341, 9.0502, 
7.4831, 6.4585, 6.4641]
R2_dksac = [0.9510, 0.9534, 0.9521, 
0.9812, 0.9892, 0.9884, 
0.9924, 0.9866, 0.9891, 
0.9925, 0.9944, 0.9944]

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
MAE_figshare = [3.0999, 3.0134, 3.0366, 
0.5103, 0.4321, 0.4074, 
3.0854, 3.0298, 3.0193, 
0.3162, 0.3166, 0.3061]
RMSE_figshare = [4.3635, 4.2672, 4.3256, 
0.6955, 0.5669, 0.5408, 
4.4044, 4.3043, 4.3027, 
0.4171, 0.4200, 0.3955]
R2_figshare = [0.6220, 0.6385, 0.6285, 
0.9904, 0.9936, 0.9942, 
0.6149, 0.6322, 0.6325, 
0.9965, 0.9965, 0.9969]

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