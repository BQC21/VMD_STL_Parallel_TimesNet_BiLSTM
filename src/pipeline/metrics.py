import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def compute_metrics(y_true, y_pred):
    """Devuelve R2, MAE, RMSE."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    R2 = r2_score(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
    return R2, MAE, RMSE