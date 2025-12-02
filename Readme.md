# STL - VMD - Parallel [TimesNet-BiLSTM]

-------------------------------------------------------------------------------------------

### Parameter configuration  
| Parameter         | Description                             | Value      |
|-------------------|-----------------------------------------|------------|
| Period            | Possible periodicity length             | 9          |
| alpha             | Bandwidth of modal functions            | 0.1        |
| k                 | Number of modes (VMD)                   | 3          |
| λ (lambda)        | Lagrange multiplier                     | 85         |
| num_kernels       | Number of kernels in TimesNet CNN block | 9          |
| num_times_blocks  | Number of TimesNet blocks                | 2          |
| top_k             | Number of significant periods            | 1          |
| dropout           | Fraction of neurons dropped             | 0.1188     |
| hidden_size       | Neurons per BiLSTM layer                | 32         |
| layers            | Number of BiLSTM layers                  | 3          |
| bidirectional     | Enable bidirectional flow                | true       |
| epochs            | Training iterations                      | 30         |
| batch_size        | Samples per batch                        | 64         |
| learning_rate     | Learning rate                            | 0.001      |

### Outputs

The proposed model improved the metrics over [2], considering MAE, RMSE and R²:


| Model                                   | MAE     | RMSE    | R²     | dataset |
|----------------------------------------|---------|---------|--------|--------|
| BiLSTM                                 | 10.2397  | 19.1686 | 0.9509 | DKASC |
| TimesNet                               | 9.3082  | 18.6095 | 0.9538 | DKASC |
| PA [TimesNet-BiLSTM]                   | 9.4747  | 18.6478 | 0.9536 | DKASC |
| STL – BiLSTM                                 | 5.9110  | 11.8351 | 0.9813 | DKASC |
| STL – TimesNet                               | 5.5962  | 9.3876 | 0.9886 | DKASC |
| STL – PA [TimesNet-BiLSTM]             | 5.5633  | 9.2110 | 0.9887 | DKASC |
| VMD – BiLSTM                                 | 3.5010  | 5.6328 | 0.9956 | DKASC |
| VMD – TimesNet                               | 4.6414  | 6.8130 | 0.9936 | DKASC |
| VMD-PA [TimesNet-BiLSTM]               | 4.0848  | 6.3868  | 0.9944 | DKASC |
| STL – VMD – BiLSTM                                 | 3.3669  | 6.1103 | 0.9949 | DKASC |
| STL – VMD – TimesNet                               | 5.1346  | 7.3156 | 0.9927 | DKASC |
| STL – VMD – PA [TimesNet-BiLSTM]      | 3.6979  | 6.4245  | 0.9943 | DKASC |
| BiLSTM                                 | 3.1638  | 4.4448 | 0.6079 | Solar station site 8 |
| TimesNet                               | 3.0838  | 4.3843 | 0.6184 | Solar station site 8 |
| PA [TimesNet-BiLSTM]                       | 3.0284  | 4,2961 | 0.6336 | Solar station site 8 |
| STL – BiLSTM                                 | 0.5038  | 0.6948 | 0.9904 | Solar station site 8 |
| STL – TimesNet                               | 0.4232  | 0.5554 | 0.9939 | Solar station site 8 |
| STL – PA [TimesNet-BiLSTM]                 | 0.4110  | 0.5472 | 0.9941 | Solar station site 8 |
| VMD – BiLSTM                                 | 3.0713  | 4.5132 | 0.5900 | Solar station site 8 |
| VMD – TimesNet                               | 3.0395  | 4.3344 | 0.6218 | Solar station site 8 |
| VMD-PA [TimesNet-BiLSTM]                   | 2.9531  | 4.2022 | 0.6445 | Solar station site 8 |
| STL – VMD – BiLSTM                                 | 0.2955  | 0.3973 | 0.9968 | Solar station site 8 |
| STL – VMD – TimesNet                               | 0.3400  | 0.4451 | 0.9960 | Solar station site 8 |
| STL – VMD – PA [TimesNet-BiLSTM] (purpose) | 0.3154  | 0.4174 | 0.9965 | Solar station site 8 |