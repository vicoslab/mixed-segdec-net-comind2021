Dataset splits for Severstaal Steel Defect Detection dataset.
# Usage

Load the splits with python pickle:

    with open(f'split_A_N.pyb', 'rb') as handle:
         train_images, test_images, validation_images = pickle.load(handle)
        
Returns sets of train/test/validation samples for given number of positive training samples (A) and number
of segmented postitve samples (N). Each samples is of form (image_name:str, is_segmented:bool)