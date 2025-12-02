import numpy as np
import pandas as pd

#################### DKASC ############################################
read_logs_dksac_BiLSTM = pd.read_excel("logs/Yulara/Predictions_BiLSTM_test.xlsx")
read_logs_dksac_TimesNet = pd.read_excel("logs/Yulara/Predictions_TimesNet_test.xlsx")
read_logs_dksac_TimesNet_BiLSTM = pd.read_excel("logs/Yulara/Predictions_TimesNet_BiLSTM_test.xlsx")
read_logs_dksac_STL_BiLSTM = pd.read_excel("logs/Yulara/Predictions_STL_BiLSTM_test.xlsx")
read_logs_dksac_STL_TimesNet = pd.read_excel("logs/Yulara/Predictions_STL_TimesNet_test.xlsx")
read_logs_dksac_STL_TimesNet_BiLSTM = pd.read_excel("logs/Yulara/Predictions_STL_TimesNet_BiLSTM_test.xlsx")
read_logs_dksac_VMD_BiLSTM = pd.read_excel("logs/Yulara/Predictions_VMD_BiLSTM_test.xlsx")
read_logs_dksac_VMD_TimesNet = pd.read_excel("logs/Yulara/Predictions_VMD_TimesNet_test.xlsx")
read_logs_dksac_VMD_TimesNet_BiLSTM = pd.read_excel("logs/Yulara/Predictions_VMD_TimesNet_BiLSTM_test.xlsx")
read_logs_dksac_STL_VMD_BiLSTM = pd.read_excel("logs/Yulara/Predictions_STL_VMD_BiLSTM_test.xlsx")
read_logs_dksac_STL_VMD_TimesNet = pd.read_excel("logs/Yulara/Predictions_STL_VMD_TimesNet_test.xlsx")
read_logs_dksac_STL_VMD_TimesNet_BiLSTM = pd.read_excel("logs/Yulara/Predictions_STL_VMD_TimesNet_BiLSTM_test.xlsx")

# True values
true_values_BiLSTM = read_logs_dksac_BiLSTM["True_Values"].values
true_values_TimesNet = read_logs_dksac_TimesNet["True_Values"].values
true_values_TimesNet_BiLSTM = read_logs_dksac_TimesNet_BiLSTM["True_Values"].values
true_values_STL_BiLSTM = read_logs_dksac_STL_BiLSTM["True_Values"].values
true_values_STL_TimesNet = read_logs_dksac_STL_TimesNet["True_Values"].values
true_values_STL_TimesNet_BiLSTM = read_logs_dksac_STL_TimesNet_BiLSTM["True_Values"].values
true_values_VMD_BiLSTM = read_logs_dksac_VMD_BiLSTM["True_Values"].values
true_values_VMD_TimesNet = read_logs_dksac_VMD_TimesNet["True_Values"].values
true_values_VMD_TimesNet_BiLSTM = read_logs_dksac_VMD_TimesNet_BiLSTM["True_Values"].values
true_values_STL_VMD_BiLSTM = read_logs_dksac_STL_VMD_BiLSTM["True_Values"].values
true_values_STL_VMD_TimesNet = read_logs_dksac_STL_VMD_TimesNet["True_Values"].values
true_values_STL_VMD_TimesNet_BiLSTM = read_logs_dksac_STL_VMD_TimesNet_BiLSTM["True_Values"].values

true_values_dksac = [true_values_BiLSTM, true_values_TimesNet, true_values_TimesNet_BiLSTM, 
true_values_STL_BiLSTM, true_values_STL_TimesNet, true_values_STL_TimesNet_BiLSTM, 
true_values_VMD_BiLSTM, true_values_VMD_TimesNet, true_values_VMD_TimesNet_BiLSTM, 
true_values_STL_VMD_BiLSTM, true_values_STL_VMD_TimesNet, true_values_STL_VMD_TimesNet_BiLSTM]

# Predicted values
predicted_values_BiLSTM = read_logs_dksac_BiLSTM["Predicted_Values"].values
predicted_values_TimesNet = read_logs_dksac_TimesNet["Predicted_Values"].values
predicted_values_TimesNet_BiLSTM = read_logs_dksac_TimesNet_BiLSTM["Predicted_Values"].values
predicted_values_STL_BiLSTM = read_logs_dksac_STL_BiLSTM["Predicted_Values"].values
predicted_values_STL_TimesNet = read_logs_dksac_STL_TimesNet["Predicted_Values"].values
predicted_values_STL_TimesNet_BiLSTM = read_logs_dksac_STL_TimesNet_BiLSTM["Predicted_Values"].values
predicted_values_VMD_BiLSTM = read_logs_dksac_VMD_BiLSTM["Predicted_Values"].values
predicted_values_VMD_TimesNet = read_logs_dksac_VMD_TimesNet["Predicted_Values"].values
predicted_values_VMD_TimesNet_BiLSTM = read_logs_dksac_VMD_TimesNet_BiLSTM["Predicted_Values"].values
predicted_values_STL_VMD_BiLSTM = read_logs_dksac_STL_VMD_BiLSTM["Predicted_Values"].values
predicted_values_STL_VMD_TimesNet = read_logs_dksac_STL_VMD_TimesNet["Predicted_Values"].values
predicted_values_STL_VMD_TimesNet_BiLSTM = read_logs_dksac_STL_VMD_TimesNet_BiLSTM["Predicted_Values"].values

predicted_values_dksac = [predicted_values_BiLSTM, predicted_values_TimesNet, predicted_values_TimesNet_BiLSTM, 
predicted_values_STL_BiLSTM, predicted_values_STL_TimesNet, predicted_values_STL_TimesNet_BiLSTM, 
predicted_values_VMD_BiLSTM, predicted_values_VMD_TimesNet, predicted_values_VMD_TimesNet_BiLSTM, 
predicted_values_STL_VMD_BiLSTM, predicted_values_STL_VMD_TimesNet, predicted_values_STL_VMD_TimesNet_BiLSTM]


#################### Figshare ############################################
read_logs_figshare_BiLSTM = pd.read_excel("logs/Figshare/Predictions_BiLSTM_test.xlsx")
read_logs_figshare_TimesNet = pd.read_excel("logs/Figshare/Predictions_TimesNet_test.xlsx")
read_logs_figshare_TimesNet_BiLSTM = pd.read_excel("logs/Figshare/Predictions_TimesNet_BiLSTM_test.xlsx")
read_logs_figshare_STL_BiLSTM = pd.read_excel("logs/Figshare/Predictions_STL_BiLSTM_test.xlsx")
read_logs_figshare_STL_TimesNet = pd.read_excel("logs/Figshare/Predictions_STL_TimesNet_test.xlsx")
read_logs_figshare_STL_TimesNet_BiLSTM = pd.read_excel("logs/Figshare/Predictions_STL_TimesNet_BiLSTM_test.xlsx")
read_logs_figshare_VMD_BiLSTM = pd.read_excel("logs/Figshare/Predictions_VMD_BiLSTM_test.xlsx")
read_logs_figshare_VMD_TimesNet = pd.read_excel("logs/Figshare/Predictions_VMD_TimesNet_test.xlsx")
read_logs_figshare_VMD_TimesNet_BiLSTM = pd.read_excel("logs/Figshare/Predictions_VMD_TimesNet_BiLSTM_test.xlsx")
read_logs_figshare_STL_VMD_BiLSTM = pd.read_excel("logs/Figshare/Predictions_STL_VMD_BiLSTM_test.xlsx")
read_logs_figshare_STL_VMD_TimesNet = pd.read_excel("logs/Figshare/Predictions_STL_VMD_TimesNet_test.xlsx")
read_logs_figshare_STL_VMD_TimesNet_BiLSTM = pd.read_excel("logs/Figshare/Predictions_STL_VMD_TimesNet_BiLSTM_test.xlsx")

# True values
true_values_BiLSTM = read_logs_figshare_BiLSTM["True_Values"].values
true_values_TimesNet = read_logs_figshare_TimesNet["True_Values"].values
true_values_TimesNet_BiLSTM = read_logs_figshare_TimesNet_BiLSTM["True_Values"].values
true_values_STL_BiLSTM = read_logs_figshare_STL_BiLSTM["True_Values"].values
true_values_STL_TimesNet = read_logs_figshare_STL_TimesNet["True_Values"].values
true_values_STL_TimesNet_BiLSTM = read_logs_figshare_STL_TimesNet_BiLSTM["True_Values"].values
true_values_VMD_BiLSTM = read_logs_figshare_VMD_BiLSTM["True_Values"].values
true_values_VMD_TimesNet = read_logs_figshare_VMD_TimesNet["True_Values"].values
true_values_VMD_TimesNet_BiLSTM = read_logs_figshare_VMD_TimesNet_BiLSTM["True_Values"].values
true_values_STL_VMD_BiLSTM = read_logs_figshare_STL_VMD_BiLSTM["True_Values"].values
true_values_STL_VMD_TimesNet = read_logs_figshare_STL_VMD_TimesNet["True_Values"].values
true_values_STL_VMD_TimesNet_BiLSTM = read_logs_figshare_STL_VMD_TimesNet_BiLSTM["True_Values"].values

true_values_figshare = [true_values_BiLSTM, true_values_TimesNet, true_values_TimesNet_BiLSTM, 
true_values_STL_BiLSTM, true_values_STL_TimesNet, true_values_STL_TimesNet_BiLSTM, 
true_values_VMD_BiLSTM, true_values_VMD_TimesNet, true_values_VMD_TimesNet_BiLSTM, 
true_values_STL_VMD_BiLSTM, true_values_STL_VMD_TimesNet, true_values_STL_VMD_TimesNet_BiLSTM]

# Predicted values
predicted_values_BiLSTM = read_logs_figshare_BiLSTM["Predicted_Values"].values
predicted_values_TimesNet = read_logs_figshare_TimesNet["Predicted_Values"].values
predicted_values_TimesNet_BiLSTM = read_logs_figshare_TimesNet_BiLSTM["Predicted_Values"].values
predicted_values_STL_BiLSTM = read_logs_figshare_STL_BiLSTM["Predicted_Values"].values
predicted_values_STL_TimesNet = read_logs_figshare_STL_TimesNet["Predicted_Values"].values
predicted_values_STL_TimesNet_BiLSTM = read_logs_figshare_STL_TimesNet_BiLSTM["Predicted_Values"].values
predicted_values_VMD_BiLSTM = read_logs_figshare_VMD_BiLSTM["Predicted_Values"].values
predicted_values_VMD_TimesNet = read_logs_figshare_VMD_TimesNet["Predicted_Values"].values
predicted_values_VMD_TimesNet_BiLSTM = read_logs_figshare_VMD_TimesNet_BiLSTM["Predicted_Values"].values
predicted_values_STL_VMD_BiLSTM = read_logs_figshare_STL_VMD_BiLSTM["Predicted_Values"].values
predicted_values_STL_VMD_TimesNet = read_logs_figshare_STL_VMD_TimesNet["Predicted_Values"].values
predicted_values_STL_VMD_TimesNet_BiLSTM = read_logs_figshare_STL_VMD_TimesNet_BiLSTM["Predicted_Values"].values

predicted_values_figshare = [predicted_values_BiLSTM, predicted_values_TimesNet, predicted_values_TimesNet_BiLSTM, 
predicted_values_STL_BiLSTM, predicted_values_STL_TimesNet, predicted_values_STL_TimesNet_BiLSTM, 
predicted_values_VMD_BiLSTM, predicted_values_VMD_TimesNet, predicted_values_VMD_TimesNet_BiLSTM, 
predicted_values_STL_VMD_BiLSTM, predicted_values_STL_VMD_TimesNet, predicted_values_STL_VMD_TimesNet_BiLSTM]