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

## Test Evaluation

|               | False       | True        |
| -----------   | ----------- | ----------- |
| **Positive**  | 17          | 1477        |
| **Negative**  | 6           | 195         |

## Test outputs

![test_output](./test_outputs/0.027_result_154.jpg)
![test_output](./test_outputs/0.405_result_1628.jpg)
![test_output](./test_outputs/0.453_result_19.jpg)
![test_output](./test_outputs/0.670_result_1467.jpg)
![test_output](./test_outputs/0.747_result_69.jpg)
![test_output](./test_outputs/0.876_result_2040.jpg)
![test_output](./test_outputs/0.939_result_2479.jpg)
![test_output](./test_outputs/0.996_result_2587.jpg)
![test_output](./test_outputs/1.000_result_2717.jpg)


## ROC

![ROC](./ROC.png)

## Precision Recall

![ROC](./precision-recall.png)

## Losses

### Loss Segmentation
![loss_seg](./loss_seg.png)

### Loss Decision
![loss_dec](./loss_dec.png)

### Total Loss
![loss](./loss.png)

### Validation
![loss_val](./loss_val.png)

### Dice and Jaccard
![dice_jaccard](./dice_jaccard.png)

## Dices

### Threshold = 0.29589 (From validation)

|             | mean        | std         |
| ----------- | ----------- | ----------- |
| **Dice**    | 0.55695     | 0.26256     |
| **Jaccard** | 0.43225     | 0.26048     |

![dice_output](./dices/0.000_dice_49.png)
![dice_output](./dices/0.000_dice_70.png)
![dice_output](./dices/0.150_dice_1662.png)
![dice_output](./dices/0.183_dice_1921.png)
![dice_output](./dices/0.190_dice_2779.png)
![dice_output](./dices/0.487_dice_2313.png)
![dice_output](./dices/0.556_dice_1511.png)
![dice_output](./dices/0.627_dice_1850.png)
![dice_output](./dices/0.730_dice_1947.png)
![dice_output](./dices/0.868_dice_2710.png)

Output of model learning is [here](./nohup.out).