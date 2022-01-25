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

# ROC

![ROC](./ROC.png)

# Precision Recall

![ROC](./precision-recall.png)


## Dices

![dice_output](./dices/0.000_dice_69.png)
![dice_output](./dices/0.007_dice_1723.png)
![dice_output](./dices/0.060_dice_2176.png)
![dice_output](./dices/0.061_dice_2371.png)
![dice_output](./dices/0.086_dice_2149.png)
![dice_output](./dices/0.088_dice_2707.png)
![dice_output](./dices/0.155_dice_2293.png)
![dice_output](./dices/0.179_dice_2411.png)
![dice_output](./dices/0.284_dice_2493.png)
![dice_output](./dices/0.289_dice_2276.png)

Output of model learning is [here](./nohup.out).