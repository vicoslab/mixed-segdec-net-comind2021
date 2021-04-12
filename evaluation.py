import os
import numpy as np
import pandas as pd
import utils
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import shutil
import pickle


def get_performance_eval(P, Y, names, data_dir, output_dir, folds, prefix='', thresholds_tups=None, save=True):
    metrics = {}
    precision_, recall_, thresholds = precision_recall_curve(Y.astype(np.int32), P)
    metrics['precision'] = precision_
    metrics['recall'] = recall_
    metrics['thresholds'] = thresholds

    FPR, TPR, _ = roc_curve(Y.astype(np.int32), P)
    AUC = auc(FPR, TPR)
    AP = average_precision_score(Y.astype(np.int32), P)
    metrics['FPR'] = FPR
    metrics['TPR'] = TPR
    metrics['AUC'] = AUC

    f_measures = 2 * (precision_ * recall_) / (precision_ + recall_ + 0.0000000001)
    metrics['f_measures'] = f_measures

    thresholds_metrics = {}
    ix_best = np.argmax(f_measures)
    if ix_best > 0:
        best_threshold = (thresholds[ix_best] + thresholds[ix_best - 1]) / 2
    else:
        best_threshold = thresholds[ix_best]
    fn0_threshold = thresholds[np.where(recall_ >= 1)][0]

    for thresh, name, dir in zip([best_threshold, fn0_threshold, 0.5], ['best', 'fn0', '50_perc'], ['best', 'fn0', '50_perc']):
        FN, FP, TN, TP = get_and_copy_falses(P, Y, thresh, data_dir, folds, names, os.path.join(output_dir, dir), prefix, save=save)
        F_measure = (2 * TP.sum()) / float(2 * TP.sum() + FP.sum() + FN.sum())
        thresholds_metrics[name] = {}
        thresholds_metrics[name]['value'] = thresh
        thresholds_metrics[name]['TP'] = TP.sum()
        thresholds_metrics[name]['TN'] = TN.sum()
        thresholds_metrics[name]['FP'] = FP.sum()
        thresholds_metrics[name]['FN'] = FN.sum()
        thresholds_metrics[name]['F_measure'] = F_measure

    metrics['AP'] = AP
    metrics['thresholds'] = thresholds_metrics

    if save:
        for thr, d in metrics['thresholds'].items():
            print(f'THRESHOLD {prefix} {thr:>15} => VALUE={d["value"]:.4f}, FP={d["FP"]} FN={d["FN"]}, AP={AP}')

    return metrics


def get_and_copy_falses(P, Y, best_threshold, data_dir, folds, names, output_dir, prefix, save=True):
    FP, FN, TN, TP = utils.calc_confusion_mat(P >= best_threshold, Y)
    # find FN and FP examples and copy them to folders
    if save:
        if not os.path.exists(output_dir):
            utils.create_folder(output_dir)
        FN_names = [(folds[i], names[i] + ".jpg") for i in range(0, len(names)) if FN[i]]
        FP_names = [(folds[i], names[i] + ".jpg") for i in range(0, len(names)) if FP[i]]
        copy_falses(FN_names, data_dir, output_dir, prefix)
        copy_falses(FP_names, data_dir, output_dir, prefix, is_FN=False)
    return FN, FP, TN, TP


def copy_falses(names, data_dir, output_dir, prefix, is_FN=True):
    for fold, n in names:
        outputs_folder = os.path.join(data_dir, f'FOLD_{fold}', prefix + 'outputs')
        f_name = list(filter(lambda s: s.endswith(n), os.listdir(outputs_folder)))
        if len(f_name) > 0:
            f_name = f_name[0]
            acc = f_name[:5]
            src_file = os.path.join(outputs_folder, f_name)
            dst_file = os.path.join(output_dir, f'{"FN" if is_FN else "FP"}_{acc}_{f_name[6:]}')

            try:
                shutil.copy(src_file, dst_file)
            except:
                print(f"error: cannot copy file {n}")


def evaluate_decision(run_dir, folds, ground_truth, img_names, predictions, prefix='', output_dir=None, thresholds=None, save=True):
    if output_dir is None:
        output_dir = run_dir

    metrics = get_performance_eval(predictions, ground_truth, img_names, run_dir, output_dir, folds, prefix=prefix, thresholds_tups=thresholds, save=save)

    best_tr_metrics = metrics['thresholds']['best']
    tp_sum = best_tr_metrics['TP']
    fp_sum = best_tr_metrics['FP']
    fn_sum = best_tr_metrics['FN']
    tn_sum = best_tr_metrics['TN']
    AP = metrics['AP']
    fp_0fn_sum = metrics['thresholds']['fn0']['FP']

    if save:
        print(f"AP: {AP:.03f}, FP/FN: {fp_sum:d}/{fn_sum:d}, FP@FN=0: {fp_0fn_sum:d}")

        with open(os.path.join(output_dir, prefix + 'accuracy.txt'), 'w') as f:
            f.write(f"TP= {tp_sum}\tFP={fp_sum}\n")
            f.write(f"FN= {fn_sum}\tTN={tn_sum}")

        with open(os.path.join(output_dir, f'{prefix}metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)
            f.close()

    return metrics


def evaluate_fold(results_folder, t_folds, t_gt, t_img_names, t_preds):
    m_test = evaluate_decision(results_folder, t_folds, t_gt, t_img_names, t_preds, prefix='', output_dir=os.path.join(results_folder), save=False)

    thresholds = m_test['thresholds']
    best = thresholds['best']
    t50 = thresholds['50_perc']
    fn0 = thresholds['fn0']

    cls_acc = (best["TP"] + best["TN"]) / (best["TP"] + best["TN"] + best["FP"] + best["FN"])
    cls_acc_50 = (t50["TP"] + t50["TN"]) / (t50["TP"] + t50["TN"] + t50["FP"] + t50["FN"])

    tpr = best["TP"] / (best["TP"] + best["FN"])
    tnr = best["TN"] / (best["TN"] + best["FP"])

    eval_res = {"ap": (m_test['AP']),
                "auc": (m_test['AUC']),
                "fps": (best['FP']),
                "fns": (best['FN']),
                "best_t": (best['value']),
                "t50_fps": (t50['FP']),
                "t50_fns": (t50['FN']),
                "fn0s": (fn0['FP']),
                "fn0_t": (fn0['value']),
                "f_measure": (best["F_measure"]),
                "cls_acc": cls_acc,
                "f_measure_50": t50["F_measure"],
                "cls_acc_50": cls_acc_50,
                "tpr": tpr,
                "tnr": tnr}
    return eval_res


def read_predictions(fold, prefix, run_dir):
    predictions, decisions, ground_truth, img_names, folds = [], [], [], [], []
    if fold is not None:
        fold_path = os.path.join(run_dir, 'FOLD_{}'.format(fold), prefix + 'results.csv')
        decisions, folds, ground_truth, img_names, predictions = read_directory(decisions, fold, fold_path, folds, ground_truth, img_names, predictions)
    else:
        results_path = os.path.join(run_dir, prefix + 'results.csv')
        decisions, folds, ground_truth, img_names, predictions = read_directory(decisions, 0, results_path, folds, ground_truth, img_names, predictions)
    img_names = list(map(str, img_names))
    predictions, decisions, ground_truth, img_names, folds = list(map(np.array, [predictions, decisions, ground_truth, img_names, folds]))

    valid_idx = (img_names != 'kos21_Part7')
    predictions = predictions[valid_idx]
    decisions = decisions[valid_idx]
    ground_truth = ground_truth[valid_idx]
    img_names = img_names[valid_idx]
    folds = folds[valid_idx]

    return decisions, folds, ground_truth, img_names, predictions


def read_directory(decisions, f, fold_path, folds, ground_truth, img_names, predictions):
    csv = pd.read_csv(fold_path)
    n_samples_in_fold = len(list(csv['prediction']))
    predictions = predictions + list(csv['prediction'])
    decisions = decisions + list(csv['decision'])
    ground_truth = ground_truth + list(csv['ground_truth'])
    img_names = img_names + list(csv['img_name'])
    folds = folds + ([f] * n_samples_in_fold)
    return decisions, folds, ground_truth, img_names, predictions
