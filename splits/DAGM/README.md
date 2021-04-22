Dataset splits for DAGM dataset.

# Usage

Load the splits with python pickle:

    with open(f'split_weakly_{N_FULLY_LABELED}.pyb', 'rb') as handle:
        all_train_images, all_test_images = pickle.load(handle)
        train_images, test_images = all_train_images[class_ix], all_test_images[class_ix]
        
Returns train/test lists, where each list entry is a set of train/test samples for that class and for a given number of fully labeled training samples (N_FULLY_LABELED).
Each sample is of form (image_name:str, is_fully_labeled:bool)