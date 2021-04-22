Dataset splits for KolektorSDD dataset.

# Usage

Load the splits with python pickle:

    with open(f'split_{N_POSITIVE}_{N_FULLY_LABELED}.pyb', 'rb') as handle:
        all_train_images, all_test_images = pickle.load(handle)
        train_images, test_images = all_train_images[fold_ix], all_test_images[fold_ix]
        
Returns train/test list, each containing train/test samples for all folds for a given number of positive training samples (N_POSITIVE) and for a given number of fully labeled training samples (N_FULLY_LABELED).
Each sample is of form (part_name:str, is_fully_labeled:bool)