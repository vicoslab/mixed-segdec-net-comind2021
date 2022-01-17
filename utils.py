import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import pandas as pd
import os
import errno
import pickle
import cv2


def create_folder(folder, exist_ok=True):
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST or not exist_ok:
            raise


def calc_confusion_mat(D, Y):
    FP = (D != Y) & (Y.astype(np.bool) == False)
    FN = (D != Y) & (Y.astype(np.bool) == True)
    TN = (D == Y) & (Y.astype(np.bool) == False)
    TP = (D == Y) & (Y.astype(np.bool) == True)

    return FP, FN, TN, TP


def plot_sample(image_name, image, segmentation, label, save_dir, decision=None, blur=True, plot_seg=False):
    plt.figure()
    plt.clf()
    plt.subplot(1, 4, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title('Input image')
    if image.shape[0] < image.shape[1]:
        image = np.transpose(image, axes=[1, 0, 2])
        segmentation = np.transpose(segmentation)
        label = np.transpose(label)
    if image.shape[2] == 1:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    plt.subplot(1, 4, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title('Groundtruth')
    plt.imshow(label, cmap="gray")

    plt.subplot(1, 4, 3)
    plt.xticks([])
    plt.yticks([])
    if decision is None:
        plt.title('Output')
    else:
        plt.title(f"Output: {decision:.5f}")
    # display max
    vmax_value = max(1, np.max(segmentation))
    plt.imshow(segmentation, cmap="jet", vmax=vmax_value)

    plt.subplot(1, 4, 4)
    plt.xticks([])
    plt.yticks([])
    plt.title('Output scaled')
    if blur:
        normed = segmentation / segmentation.max()
        blured = cv2.blur(normed, (32, 32))
        plt.imshow((blured / blured.max() * 255).astype(np.uint8), cmap="jet")
    else:
        plt.imshow((segmentation / segmentation.max() * 255).astype(np.uint8), cmap="jet")

    out_prefix = '{:.3f}_'.format(decision) if decision is not None else ''

    plt.savefig(f"{save_dir}/{out_prefix}result_{image_name}.jpg", bbox_inches='tight', dpi=300)
    plt.close()

    if plot_seg:
        jet_seg = cv2.applyColorMap((segmentation * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f"{save_dir}/{out_prefix}_segmentation_{image_name}.png", jet_seg)


def evaluate_metrics(samples, results_path, run_name, segmentation_predicted, segmentation_truth, images):
    samples = np.array(samples)

    img_names = samples[:, 4]
    predictions = samples[:, 0]
    labels = samples[:, 3].astype(np.float32)

    metrics = get_metrics(labels, predictions)
    dice_mean, dice_std, jaccard_mean, jaccard_std = dice_jaccard(segmentation_predicted, segmentation_truth, metrics['best_thr'], images, img_names, results_path)

    df = pd.DataFrame(
        data={'prediction': predictions,
              'decision': metrics['decisions'],
              'ground_truth': labels,
              'img_name': img_names})
    df.to_csv(os.path.join(results_path, 'results.csv'), index=False)

    print(
        f'{run_name} EVAL AUC={metrics["AUC"]:f}, and AP={metrics["AP"]:f}, w/ best thr={metrics["best_thr"]:f} at f-m={metrics["best_f_measure"]:.3f} and FP={sum(metrics["FP"]):d}, FN={sum(metrics["FN"]):d}, Dice: mean: {dice_mean:f}, std: {dice_std}, Jaccard: mean: {jaccard_mean:f}, std: {jaccard_std}')

    with open(os.path.join(results_path, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
        f.close()

    plt.figure(1)
    plt.clf()
    plt.plot(metrics['recall'], metrics['precision'])
    plt.title('Average Precision=%.4f' % metrics['AP'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f"{results_path}/precision-recall.pdf", bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    plt.plot(metrics['FPR'], metrics['TPR'])
    plt.title('AUC=%.4f' % metrics['AUC'])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(f"{results_path}/ROC.pdf", bbox_inches='tight')


def get_metrics(labels, predictions):
    metrics = {}
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['thresholds'] = thresholds
    f_measures = 2 * np.multiply(recall, precision) / (recall + precision + 1e-8)
    metrics['f_measures'] = f_measures
    ix_best = np.argmax(f_measures)
    metrics['ix_best'] = ix_best
    best_f_measure = f_measures[ix_best]
    metrics['best_f_measure'] = best_f_measure
    best_thr = thresholds[ix_best]
    metrics['best_thr'] = best_thr
    FPR, TPR, _ = roc_curve(labels, predictions)
    metrics['FPR'] = FPR
    metrics['TPR'] = TPR
    AUC = auc(FPR, TPR)
    metrics['AUC'] = AUC
    AP = auc(recall, precision)
    metrics['AP'] = AP
    decisions = predictions >= best_thr
    metrics['decisions'] = decisions
    FP, FN, TN, TP = calc_confusion_mat(decisions, labels)
    metrics['FP'] = FP
    metrics['FN'] = FN
    metrics['TN'] = TN
    metrics['TP'] = TP
    metrics['accuracy'] = (sum(TP) + sum(TN)) / (sum(TP) + sum(TN) + sum(FP) + sum(FN))
    return metrics

def dice_jaccard(segmentation_predicted, segmentation_truth, threshold, images=None, image_names=None, run_path=None):

    results_dice = []
    result_jaccard = []

    # Preverimo ali so vsi listi enako dolgi
    if images is not None:
        if not (len(segmentation_predicted) == len(segmentation_truth) == len(images) == len(image_names)):
            raise ValueError('Not equal size of segmentation masks or images')
    else:
        if not (len(segmentation_predicted) == len(segmentation_truth)):
            raise ValueError('Not equal size of segmentation masks')
    
    # Save folder
    if run_path is not None:
        save_folder = f"{run_path}/dices"
        create_folder(save_folder)

    # Za vsak par izračunamo dice in jaccard
    for i in range(len(segmentation_predicted)):
        seg_pred = segmentation_predicted[i]
        seg_true_bin = segmentation_truth[i]

        # Naredimo binarne maske s ustreznim thresholdom
        seg_pred_bin = (seg_pred > threshold).astype(np.uint8)

        # Dice
        dice = (2 * (seg_true_bin * seg_pred_bin).sum() + 1e-15) / (seg_true_bin.sum() + seg_pred_bin.sum() + 1e-15)
        results_dice += [dice]

        # Jaccard
        intersection = (seg_pred_bin * seg_true_bin).sum()
        union = seg_pred_bin.sum() + seg_true_bin.sum() - intersection
        jaccard = (intersection + 1e-15) / (union + 1e-15)
        result_jaccard += [jaccard]

        # Vizualizacija
        if images is not None:
            image = images[i]
            image_name = image_names[i]
            plt.figure()
            plt.clf()

            plt.subplot(1, 4, 1)
            plt.xticks([])
            plt.yticks([])
            plt.title('Image')
            plt.imshow(image)
            plt.xlabel(f"Threshold: {round(threshold, 5)}")
            
            plt.subplot(1, 4, 2)
            plt.xticks([])
            plt.yticks([])
            plt.title('Groundtruth')
            plt.imshow(seg_true_bin, cmap='gray')
            
            plt.subplot(1, 4, 3)
            plt.xticks([])
            plt.yticks([])
            plt.title('Segmentation')
            plt.imshow(seg_pred, cmap='gray')
            plt.xlabel(f"Jaccard: {round(jaccard, 5)}")
            
            plt.subplot(1, 4, 4)
            plt.xticks([])
            plt.yticks([])
            plt.title('Segmentation\nmask')
            plt.imshow(seg_pred_bin, cmap='gray')
            plt.xlabel(f"Dice: {round(dice, 5)}")
            plt.savefig(f"{save_folder}/{round(dice, 3):.3f}_dice_{image_name}.png", bbox_inches='tight', dpi=300)
            plt.close()

    # Vrnemo povprečno vrednost ter standardno deviacijo za dice in jaccard
    return np.mean(results_dice), np.std(results_dice), np.mean(result_jaccard), np.std(result_jaccard)