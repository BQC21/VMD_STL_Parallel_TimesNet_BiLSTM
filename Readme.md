# STL - VMD - Parallel [TimesNet-BiLSTM]

-------------------------------------------------------------------------------------------

### Parameter configuration  
| Parameter         | Description                             | Value      |
|-------------------|-----------------------------------------|------------|
| Period            | Possible periodicity length             | 9          |
| alpha             | Bandwidth of modal functions            | 0.1        |
| k                 | Number of modes (VMD)                   | 3          |
| λ (lambda)        | Lagrange multiplier                     | 85         |
| num_kernels       | Number of kernels in TimesNet CNN block | 7          |
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
| TimesNet                               | 10.4000  | 19.0600 | 0.9515 | DKASC |
| PA [TimesNet-BiLSTM]                   | 9.3171  | 18.7371 | 0.9531 | DKASC |
| STL – BiLSTM                                 | 5.9110  | 11.8351 | 0.9813 | DKASC |
| STL – TimesNet                               | 6.4664  | 9.8836 | 0.9870 | DKASC |
| STL – PA [TimesNet-BiLSTM]             | 5.3263  | 9.3981 | 0.9862 | DKASC |
| VMD – BiLSTM                                 | 3.5010  | 5.6328 | 0.9956 | DKASC |
| VMD – TimesNet                               | 4.3656  | 6.5420 | 0.9941 | DKASC |
| VMD-PA [TimesNet-BiLSTM]               | 4.3025  | 6.3747  | 0.9944 | DKASC |
| STL – VMD – BiLSTM                                 | 3.3669  | 6.1103 | 0.9949 | DKASC |
| STL – VMD – TimesNet                               | 4.9964  | 7.6306 | 0.9920 | DKASC |
| STL – VMD – PA [TimesNet-BiLSTM] (purpose)      | 4.4401  | 7.0507  | 0.9932 | DKASC |
| BiLSTM                                 |   |  |  | Solar station site 8 |
| TimesNet                               |   |  |  | Solar station site 8 |
| PA [TimesNet-BiLSTM]                       |   |  |  | Solar station site 8 |
| STL – BiLSTM                                 |   |  |  | Solar station site 8 |
| STL – TimesNet                               |   |  |  | Solar station site 8 |
| STL – PA [TimesNet-BiLSTM]                 |   |  |  | Solar station site 8 |
| VMD – BiLSTM                                 |   |  |  | Solar station site 8 |
| VMD – TimesNet                               |   |  |  | Solar station site 8 |
| VMD-PA [TimesNet-BiLSTM]                   |   |  |  | Solar station site 8 |
| STL – VMD – BiLSTM                                 |   |  |  | Solar station site 8 |
| STL – VMD – TimesNet                               |   |  |  | Solar station site 8 |
| STL – VMD – PA [TimesNet-BiLSTM] (purpose) |   |  |  | Solar station site 8 |