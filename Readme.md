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
| PA [TimesNet-BiLSTM]                   | 10.1088  | 18.9476 | 0.9521 | DKASC |
| STL – PA [TimesNet-BiLSTM]             | 5.5533 | 9.3104 | 0.9884 | DKASC |
| VMD – BiLSTM                                 | 4.4302  | 7.5547 | 0.9924 | DKASC |
| VMD – TimesNet                               | 5.5409  | 10.0341 | 0.9866 | DKASC |
| VMD-PA [TimesNet-BiLSTM]               | 5.7206  | 9.0523  | 0.9891 | DKASC |
| STL – VMD – BiLSTM                                 | 4.2659  | 7.4831 | 0.9925 | DKASC |
| STL – VMD – TimesNet                               | 3.9706  | 6.4585 | 0.9944 | DKASC |
| STL – VMD – PA [TimesNet-BiLSTM]      | 3.9234  | 6.4641  | 0.9944 | DKASC |
| PA [TimesNet-BiLSTM]                       | 3.0366  | 4.3256 | 0.6285 | Solar station site 8 |
| STL – PA [TimesNet-BiLSTM]                 | 0.4074  | 0.5408 | 0.9942 | Solar station site 8 |
| VMD – BiLSTM                                 | 3.0854  | 4.4044 | 0.6149 | Solar station site 8 |
| VMD – TimesNet                               | 3.0298  | 4.3043 | 0.6322 | Solar station site 8 |
| VMD-PA [TimesNet-BiLSTM]                   | 3.0193| 4.3027 | 0.6325 | Solar station site 8 |
| STL – VMD – BiLSTM                                 | 0.3162  | 0.4171 | 0.9965 | Solar station site 8 |
| STL – VMD – TimesNet                               | 0.3166  | 0.4200 | 0.9965 | Solar station site 8 |
| STL – VMD – PA [TimesNet-BiLSTM] (purpose) | 0.3154  | 0.3061 | 0.9969 | Solar station site 8 |