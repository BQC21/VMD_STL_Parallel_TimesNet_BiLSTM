# NumPy / Pandas / Matplotlib
import seaborn as sns
import numpy as np
import pandas as pd

# Build log arrays
from build_log_arrays import true_values_dksac, predicted_values_dksac, true_values_figshare, predicted_values_figshare

#################### DKASC BOXPLOT ############################################

# Build log arrays
from build_log_arrays import true_values_dksac, predicted_values_dksac, true_values_figshare, predicted_values_figshare

# Boxplot
sns.boxplot(data=true_values_dksac, predicted_values_dksac)
plt.show()

#################### Figshare BOXPLOT ############################################

# Boxplot