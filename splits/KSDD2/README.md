Dataset splits for KolektorSDD2 dataset.

# Usage

Load the splits with python pickle:

    with open(f'split_N.pyb', 'rb') as handle:
         train_images, test_images = pickle.load(handle)
        
Returns sets of train/test samples for a given number of fully labeled training samples (N).
Each sample is of form (image_name:str, is_segmented:bool)