Dataset splits for KolektorSDD2 dataset.

# Usage

Load the splits with python pickle:

    with open(f'split_{N_FULLY_LABELED}.pyb', 'rb') as handle:
         train_images, test_images = pickle.load(handle)
        
Returns sets of train/test samples for a given number of fully labeled training samples (N_FULLY_LABELED).
Each sample s of form (image_name:str, is_fully_labeled:bool)