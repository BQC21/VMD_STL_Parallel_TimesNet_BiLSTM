# STL - VMD - Parallel [TimesNet-BiLSTM]

On this branch, doing ablation study for both datasets only consider architectures with VMD. For this case only try with different alpha values, those matching the best results for alpha_test branch.

-------------------------------------------------------------------------------------------

### Outputs

**For alpha 0.5**


| Model                                   | MAE     | RMSE    | R²     | dataset |
|----------------------------------------|---------|---------|--------|--------|
| VMD – BiLSTM                                 | 3.9446 | 6.2171 | 0.9948 | DKASC |
| VMD – TimesNet                               | 6.2041 | 9.1803 | 0.9887 | DKASC |
| VMD-PA [TimesNet-BiLSTM]               | 4.6048 | 6.6344 | 0.9941 | DKASC |
| STL – VMD – BiLSTM                                 | 4.4456 | 7.0261 | 0.9934 | DKASC |
| STL – VMD – TimesNet                               | 4.2899 | 7.2636 | 0.9930 | DKASC |
| STL – VMD – PA [TimesNet-BiLSTM] (purpose)     | 4.3452 | 6.9939 | 0.9935 | DKASC |
| VMD – BiLSTM                                 | 3.1635 | 4.4111 | 0.6136 | Solar station site 8 |
| VMD – TimesNet                               | 3.0607 | 4.3371 | 0.6265 | Solar station site 8 |
| VMD-PA [TimesNet-BiLSTM]                   | 3.0288 | 4.3184 | 0.6297 | Solar station site 8 |
| STL – VMD – BiLSTM                                 | 0.2395 | 0.3213 | 0.9980 | Solar station site 8 |
| STL – VMD – TimesNet                               | 0.3638 | 0.4839 | 0.9954 | Solar station site 8 |
| STL – VMD – PA [TimesNet-BiLSTM] (purpose) | 0.2873 | 0.3742 | 0.9972 | Solar station site 8 |

**For alpha 2000**

| Model                                   | MAE     | RMSE    | R²     | dataset |
|----------------------------------------|---------|---------|--------|--------|
| VMD – BiLSTM                                 | 2.3838 | 3.4425 | 0.9984 | DKASC |
| VMD – TimesNet                               | 2.8602 | 3.9802 | 0.9978 | DKASC |
| VMD-PA [TimesNet-BiLSTM]               | 2.1676 | 3.3143 | 0.9985 | DKASC |
| STL – VMD – BiLSTM                                 | 2.0094 | 2.7444 | 0.9990 | DKASC |
| STL – VMD – TimesNet                               | 2.5738 | 3.4981 | 0.9983 | DKASC |
| STL – VMD – PA [TimesNet-BiLSTM] (purpose)     | 2.1580 | 3.0244 | 0.9987 | DKASC |
| VMD – BiLSTM                                 | 2.9682 | 4.1099 | 0.6509 | Solar station site 8 |
| VMD – TimesNet                               | 2.9609 | 4.1835 | 0.6383 | Solar station site 8 |
| VMD-PA [TimesNet-BiLSTM]                   | 2.9466 | 4.1591 | 0.6425 | Solar station site 8 |
| STL – VMD – BiLSTM                                 | 0.3104 | 0.3953 | 0.9968 | Solar station site 8 |
| STL – VMD – TimesNet                               | 0.3463 | 0.4323 | 0.9961 | Solar station site 8 |
| STL – VMD – PA [TimesNet-BiLSTM] (purpose) | 0.3712 | 0.4783 | 0.9953 | Solar station site 8 |

