Dataset splits for Severstaal Steel Defect Detection dataset.
# Usage

Load the splits with python pickle:

    with open(f'split_{N_POSITIVE}_{N_FULLY_LABELED}.pyb', 'rb') as handle:
         train_images, test_images, validation_images = pickle.load(handle)
        
Returns sets of train/test/validation samples for given number of positive training samples (N_POSITIVE) (N_ALL in paper) and number
of fully labeled positive samples (N_FULLY_LABELED).
Each samples is of form (image_name:str, is_fully_labeled:bool)