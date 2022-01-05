# Results of training on crack_segmentation dataset

## Dataset

| Parameter         | Value       |
| -----------       | ----------- |
| Input channels    | 3           |
| Input height      | 640         |
| Input width       | 232         |
| Input width       | 232         |
| Train samples     | 7908        |
| Test samples      | 1695        |
| Validation samples| 1695        |
| Segmented samples | 6921        |


| Set         | Positives   | Negatives   |  Sum        |
| ----------- | ----------- | ----------- | ----------- |
| Train       | 6921        | 987         | 7908        |
| Test        | 1483        | 212         | 1695        |
| Validation  | 1483        | 212         | 1695        |
| **Sum**     | 9887        | 1411        | 11298       |

## Run parameters
| Parameter                      | Value       |
| -----------                    | ----------- |
| Batch size                     | 1           |
| Epochs                         | 50          |
| Learning rate                  | 1           |
| Delta CLS Loss                 | 0.01        |
| Dilate                         | 1           |
| Dynamically balanced loss      | True        |
| Frequency-of-use sampling      | True        |
| Gradien-flow adjustment        | True        |
| Weighted segmentation loss     | True        |
| Weighted segmentation loss Max | 1.0         |
| Weighted segmentation loss P   | 2.0         |
| Use best model                 | True        |
| Validate                       | True        |
| Validate on test               | False       |


## Results

|                  | True        | False       |
| -----------      | ----------- | ----------- |
| **Positives**    | 1473        | 10          |
| **Negatives**    | 179         | 33          |


![AUC](./ROC.jpg)
![Ap](./precision-recall.jpg)
![Loss Val](./loss_val.png)
![Loss](./loss.png)
![Loss dec](./loss_dec.png)

## Test outputs

![test_output](./test_outputs/0.012_result_24.jpg)
![test_output](./test_outputs/0.381_result_2515.jpg)
![test_output](./test_outputs/0.839_result_2074.jpg)
![test_output](./test_outputs/0.967_result_2598.jpg)
![test_output](./test_outputs/0.999_result_1738.jpg)

Output of model learning is [here](./nohup.out).