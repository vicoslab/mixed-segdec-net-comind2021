import os, random, shutil

random.seed(420)

DATASET = 'crack_segmentation'
FACTOR = 10

path_datasets = './datasets'
path_subset = os.path.join('./datasets', DATASET + '_subset')
dirs = ['train_negative', 'train_positive', 'val_negative', 'val_positive', 'test_negative', 'test_positive']

def copy_samples(samples, path_from, path_to):
    i = 0
    for sample in samples:
        
        sample_GT = sample.split('.')[0] + '_GT.jpg'
        sample_path_from = os.path.join(path_from, sample)
        sample_path_to = os.path.join(path_to, sample)
        sample_GT_path_from = os.path.join(path_from, sample_GT)
        sample_GT_path_to = os.path.join(path_to, sample_GT)
        
        shutil.copy2(sample_path_from, sample_path_to)
        shutil.copy2(sample_GT_path_from, sample_GT_path_to)
        
        i += 1
    
    print(f'Copied {i} samples and their GT from {path_from} to {path_to}')

for dir in dirs:
    path = os.path.join('./datasets', DATASET, dir)
    new_path = os.path.join(path_subset, dir)
    samples = [i for i in sorted(os.listdir(path)) if 'GT' not in i]
    number_of_selected_samples = len(samples) // FACTOR
    random_samples = random.sample(samples, number_of_selected_samples)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    copy_samples(random_samples, path, new_path)