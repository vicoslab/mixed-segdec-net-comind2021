import argparse
import os
import evaluation


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--RESULTS_PATH', type=str, required=True)
    parser.add_argument('--RUN_NAME', type=str, required=True)
    parser.add_argument('--DATASET', type=str, required=True)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    results_path = args.RESULTS_PATH
    run_name = args.RUN_NAME
    dataset = args.DATASET

    folds_dict = {"KSDD": [0, 1, 2]}

    results_folder = os.path.join(results_path, dataset, run_name)

    print(f'Running evaluation for RUN {results_folder}')

    folds = folds_dict[dataset]

    ap, auc, fps, fns, best_t, t50_fps, t50_fns, fn0s, fn0_t = 0, 0, 0, 0, -1, 0, 0, 0, -1
    for f in folds:
        t_dec, t_folds, t_gt, t_img_names, t_preds = evaluation.read_predictions(f, '', results_folder)
        fold_eval_res = evaluation.evaluate_fold(results_folder, t_folds, t_gt, t_img_names, t_preds)
        ap += fold_eval_res["ap"]
        auc += fold_eval_res["auc"]
        fps += fold_eval_res["fps"]
        fns += fold_eval_res["fns"]
        t50_fps += fold_eval_res["t50_fps"]
        t50_fns += fold_eval_res["t50_fns"]
        fn0s += fold_eval_res["fn0s"]
    ap /= len(folds)
    auc /= len(folds)

    print(f"RUN {run_name}: AP:{ap:.5f}, AUC:{auc:.5f}, FP={fps}, FN={fns}, FN@.5={t50_fns}, FP@.5={t50_fps}, FP@FN0={fn0s}")
