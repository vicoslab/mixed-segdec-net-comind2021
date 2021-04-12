from end2end import End2End
import argparse
from config import Config


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--GPU', type=int, required=True, help="ID of GPU used for training/evaluation.")
    parser.add_argument('--RUN_NAME', type=str, required=True, help="Name of the run, used as directory name for storing results.")
    parser.add_argument('--DATASET', type=str, required=True, help="Which dataset to use.")
    parser.add_argument('--DATASET_PATH', type=str, required=True, help="Path to the dataset.")

    parser.add_argument('--EPOCHS', type=int, required=True, help="Number of training epochs.")

    parser.add_argument('--LEARNING_RATE', type=float, required=True, help="Learning rate.")
    parser.add_argument('--DELTA_CLS_LOSS', type=float, required=True, help="Weight delta for classification loss.")

    parser.add_argument('--BATCH_SIZE', type=int, required=True, help="Batch size for training.")

    parser.add_argument('--WEIGHTED_SEG_LOSS', type=str2bool, required=True, help="Whether to use weighted segmentation loss.")
    parser.add_argument('--WEIGHTED_SEG_LOSS_P', type=float, required=False, default=None, help="Degree of polynomial for weighted segmentation loss.")
    parser.add_argument('--WEIGHTED_SEG_LOSS_MAX', type=float, required=False, default=None, help="Scaling factor for weighted segmentation loss.")
    parser.add_argument('--DYN_BALANCED_LOSS', type=str2bool, required=True, help="Whether to use dynamically balanced loss.")
    parser.add_argument('--GRADIENT_ADJUSTMENT', type=str2bool, required=True, help="Whether to use gradient adjustment.")
    parser.add_argument('--FREQUENCY_SAMPLING', type=str2bool, required=False, help="Whether to use frequency-of-use based sampling.")

    parser.add_argument('--DILATE', type=int, required=False, default=None, help="Size of dilation kernel for labels")

    parser.add_argument('--FOLD', type=int, default=None, help="Which fold (KSDD) or class (DAGM) to train.")
    parser.add_argument('--TRAIN_NUM', type=int, default=None, help="Number of positive training samples for KSDD or STEEL.")
    parser.add_argument('--NUM_SEGMENTED', type=int, required=True, default=None, help="Number of segmented positive  samples.")
    parser.add_argument('--RESULTS_PATH', type=str, default=None, help="Directory to which results are saved.")

    parser.add_argument('--VALIDATE', type=str2bool, default=None, help="Whether to validate during training.")
    parser.add_argument('--VALIDATE_ON_TEST', type=str2bool, default=None, help="Whether to validate on test set.")
    parser.add_argument('--VALIDATION_N_EPOCHS', type=int, default=None, help="Number of epochs between consecutive validation runs.")
    parser.add_argument('--USE_BEST_MODEL', type=str2bool, default=None, help="Whether to use the best model according to validation metrics for evaluation.")

    parser.add_argument('--ON_DEMAND_READ', type=str2bool, default=None, help="Whether to use on-demand read of data from disk instead of storing it in memory.")
    parser.add_argument('--REPRODUCIBLE_RUN', type=str2bool, default=None, help="Whether to fix seeds and disable CUDA benchmark mode.")

    parser.add_argument('--MEMORY_FIT', type=int, default=None, help="How many images can be fitted in GPU memory.")
    parser.add_argument('--SAVE_IMAGES', type=str2bool, default=None, help="Save test images or not.")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    configuration = Config()
    configuration.merge_from_args(args)
    configuration.init_extra()

    end2end = End2End(cfg=configuration)
    end2end.train()
