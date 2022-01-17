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

## Dice test outputs

![test_output](dices\0.0_dice_113.png)
![test_output](dices\0.12_dice_2523.png)
![test_output](dices\0.217_dice_2267.png)
![test_output](dices\0.665_dice_2484.png)
![test_output](dices\0.737_dice_1962.png)
![test_output](dices\0.897_dice_2368.png)
![test_output](dices\1.0_dice_165.png)