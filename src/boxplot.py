# NumPy / Pandas / Matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

################

# boxplot for DKSAC
boxplot_dksac = plt.boxplot(DKASC["MAE"], DKASC["RMSE"], DKASC["R2"])
plt.show()

# boxplot for figshare
boxplot_figshare = plt.boxplot(figshare["MAE"], figshare["RMSE"], figshare["R2"])
plt.show()