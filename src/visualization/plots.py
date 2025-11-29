import matplotlib.pyplot as plt
import numpy as np

def plot_losses(loss_train, loss_valid, imf_col=None):
    plt.figure(figsize=(8,4))
    plt.plot(loss_train, label='train_loss')
    plt.plot(loss_valid, label='valid_loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.title(f'Evolution of the loss function ({imf_col})')
    plt.legend(); plt.tight_layout()
    plt.show()

def visualize(days, tt_inv, tp_inv, TARGET):
    plt.figure(figsize=(12,4))
    plt.plot(np.linspace(1,days*47,days*47), tt_inv[300:300+days*47], label='y_true')
    plt.plot(np.linspace(1,days*47,days*47), tp_inv[300:300+days*47], label='y_pred')
    plt.title('Testing'); plt.xlabel('samples')
    plt.ylabel(TARGET); plt.legend(); plt.tight_layout()
    plt.show()

def scatter(tt_inv, tp_inv):
    plt.figure(figsize=(5,5))
    plt.scatter(tt_inv, tp_inv, s=5, alpha=0.5)
    plt.title('Scatter plot: y_true vs y_pred')
    plt.xlabel('y_true'); plt.ylabel('y_pred')
    lo = float(np.nanmin([tt_inv.min(), tp_inv.min()]))
    hi = float(np.nanmax([tt_inv.max(), tp_inv.max()]))
    xs = np.linspace(lo, hi, 100)
    plt.plot(xs, xs, linewidth=1)
    plt.tight_layout()
    plt.show()