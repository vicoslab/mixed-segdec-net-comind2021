import os
import evaluation
import pandas as pd
from operator import itemgetter
from config import load_from_dict

def get_params(cfg):
    params_lines = []
    for k, v in sorted(cfg.get_as_dict().items(), key=lambda x: x[0]):
        params_lines.append(f'{k}:{v}')
    return ','.join(params_lines)


def get_run_config(run_path, fold):
    params = {}
    if fold is not None:
        params_file = os.path.join(run_path, f'FOLD_{fold}', 'run_params.txt')
    else:
        params_file = os.path.join(run_path, 'run_params.txt')
    with open(params_file, 'r') as f:
        for l in f.readlines():
            k, v = l.split(':')
            params[k] = v.strip()
    return load_from_dict(params)


def read_results(results_path, dataset, folds=None, dagm_join=False, sortkey=itemgetter(0)):
    results = []
    results_columns = ['RUN_NAME',
                       'TN', "N_SEG",
                       'W_SEG_LOSS', 'W_P', 'W_MAX',
                       'FRQ_SMP', 'DYN_B_L', 'DELTA',
                       'EPS', 'LR',
                       'AUC', 'AP',
                       'FP', 'FN', 'FALSES', 'THRESH',
                       "F_MSR", "CLS_ACC", "TPR", "TNR",
                       '50_FP', '50_FN', '50_FALSES', '50_FMS', '50_CA',
                       'FP@FN=0', 'THRESH@FN=0',
                       'PATH', 'CONFIGURATION'
                       ]
    if dataset == "DAGM" and not dagm_join:
        for f in folds:
            process_dataset(results_path, dataset, [f], results, dagm_join)
    else:
        process_dataset(results_path, dataset, folds, results, dagm_join)

    results = sorted(results, key=sortkey)
    df = pd.DataFrame(results, columns=results_columns)
    return df


def process_dataset(results_path, dataset, folds, results, dagm_join):
    for run_name in os.listdir(os.path.join(results_path, dataset)):
        run_path = os.path.join(results_path, dataset, run_name)
        try:
            print(f"Processing run_path: {run_path}")
            cfg = get_run_config(run_path, None if folds is None else folds[0])
            ap, auc, fps, fns, t50_fps, t50_fns, fn0s, f_measure, cls_acc, f_measure_50, cls_acc_50, tpr, tnr = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            best_t, fn0_t = -1, -1
            for f in folds:
                t_dec, t_folds, t_gt, t_img_names, t_preds = evaluation.read_predictions(f, '', run_path)
                fold_eval_res = evaluation.evaluate_fold(run_path, t_folds, t_gt, t_img_names, t_preds)
                if len(folds) == 1:
                    best_t = fold_eval_res["best_t"]
                    fn0_t = fold_eval_res["fn0_t"]
                ap += fold_eval_res["ap"]
                auc += fold_eval_res["auc"]
                fps += fold_eval_res["fps"]
                fns += fold_eval_res["fns"]
                t50_fps += fold_eval_res["t50_fps"]
                t50_fns += fold_eval_res["t50_fns"]
                fn0s += fold_eval_res["fn0s"]
                f_measure += fold_eval_res["f_measure"]
                cls_acc += fold_eval_res["cls_acc"]
                f_measure_50 += fold_eval_res["f_measure_50"]
                cls_acc_50 += fold_eval_res["cls_acc_50"]
                tpr += fold_eval_res["tpr"]
                tnr += fold_eval_res["tnr"]
            ap /= len(folds)
            auc /= len(folds)
            f_measure /= len(folds)
            cls_acc /= len(folds)
            f_measure_50 /= len(folds)
            cls_acc_50 /= len(folds)
            tpr /= len(folds)
            tnr /= len(folds)

            if dataset == "DAGM" and not dagm_join:
                run_name = f"{run_name}_FOLD_{folds[0]}"

            results.append(
                [run_name,
                 cfg.TRAIN_NUM, cfg.NUM_SEGMENTED,
                 cfg.WEIGHTED_SEG_LOSS, cfg.WEIGHTED_SEG_LOSS_P, cfg.WEIGHTED_SEG_LOSS_MAX,
                 cfg.FREQUENCY_SAMPLING, cfg.DYN_BALANCED_LOSS, cfg.DELTA_CLS_LOSS,
                 cfg.EPOCHS, cfg.LEARNING_RATE,
                 f"{auc:.5f}", f"{ap:.5f}",
                 fps, fns, fps + fns, f"{best_t:.5f}",
                 f"{f_measure:.5f}", f"{cls_acc:.5f}", f"{tpr:.5f}", f"{tnr:.5f}",
                 t50_fps, t50_fns, t50_fps + t50_fns, f"{f_measure_50:.5f}", f"{cls_acc_50:.5f}",
                 fn0s, f"{fn0_t:.5f}",
                 run_path, get_params(cfg)]
            )

        except Exception as f:
            print(f'Error reading RUN {run_path} with Exception {f} ')


def main():
    # dataset,results_folder = "STEEL", '/home/jakob/outputs/WEAKLY_LABELED/STEEL/GRADIENT'
    # dataset, results_folder = "KSDD2", '/home/jakob/outputs/WEAKLY_LABELED/KSDD2/GRADIENT'
    # dataset, results_folder = "DAGM", '/home/jakob/outputs/WEAKLY_LABELED/DAGM/GS'
    dataset, results_folder = "KSDD", '/home/jakob/outputs/WEAKLY_LABELED/RELEASE/'

    dagm_join = False # If True will join(average) results for all classes

    folds_dict = {
        'KSDD': [0, 1, 2],
        'KSDD2': [None],
        'STEEL': [None],
        'DAGM': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    results = read_results(results_folder, dataset, folds_dict[dataset], dagm_join, sortkey=itemgetter(0))
    results.to_csv(os.path.join(results_folder, f'{dataset}_summary{f"_joined" if dataset == "DAGM" and dagm_join else ""}.csv'), index=False)


if __name__ == '__main__':
    main()
