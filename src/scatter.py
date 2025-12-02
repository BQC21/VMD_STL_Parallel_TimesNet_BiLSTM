# NumPy / Pandas / Matplotlib
from matplotlib.axes._axes import Axes

from pandas._typing import ArrayLike

from matplotlib.axes import Axes
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Build log arrays
from build_log_arrays import true_values_dksac, predicted_values_dksac, true_values_figshare, predicted_values_figshare

# There are 12 subplots, here only one dataset is used.
# You probably want to plot different predictions in each axis.
# For now, just plot the same in each, but plot onto the axis, not plt.figure.
def scatter_on_ax(ax: Axes, tt_inv: Any, tp_inv: Any):
    tt_arr = np.asarray(tt_inv, dtype=float)
    tp_arr = np.asarray(tp_inv, dtype=float)
    ax.scatter(tt_arr, tp_arr, s=5, alpha=0.5)
    ax.set_title('Scatter plot: y_true vs y_pred')
    ax.set_xlabel('y_true')
    ax.set_ylabel('y_pred')
    lo = float(np.nanmin([tt_arr.min(), tp_arr.min()]))
    hi = float(np.nanmax([tt_arr.max(), tp_arr.max()]))
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, xs, linewidth=1)

#################### DKASC SCATTER ############################################

fig_dksac, axes_dksac = plt.subplots(4, 3, figsize=(15, 10))
fig_dksac.suptitle('Scatter plot: y_true vs y_pred for DKSAC', fontsize=16)

# axes_dksac is a 2D numpy array by default, flatten it properly
# Only flatten if it's an ndarray (not a single Axes object)
if isinstance(axes_dksac, np.ndarray):
    axes_dksac = axes_dksac.flatten()
else:
    axes_dksac = [axes_dksac]

# Plot each model’s true/pred pair on its own axis
for ax, tt_inv, tp_inv in zip(axes_dksac, true_values_dksac, predicted_values_dksac):
    scatter_on_ax(ax, tt_inv, tp_inv)

plt.tight_layout()
fig_dksac.savefig('scatter_dksac.png')


#################### Figshare SCATTER ############################################

fig_figshare, axes_figshare = plt.subplots(4, 3, figsize=(15, 10))
fig_figshare.suptitle('Scatter plot: y_true vs y_pred for Figshare', fontsize=16)

# axes_figshare is a 2D numpy array by default, flatten it properly
# Only flatten if it's an ndarray (not a single Axes object)
if isinstance(axes_figshare, np.ndarray):
    axes_figshare = axes_figshare.flatten()
else:
    axes_figshare = [axes_figshare]

for ax, tt_inv, tp_inv in zip(axes_figshare, true_values_figshare, predicted_values_figshare):
    scatter_on_ax(ax, tt_inv, tp_inv)

plt.tight_layout()
fig_figshare.savefig('scatter_figshare.png')
plt.show()