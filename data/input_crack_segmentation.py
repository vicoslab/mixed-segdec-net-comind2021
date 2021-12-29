import os
import numpy as np
from data.dataset import Dataset
from config import Config

class CrackSegmentationDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(CrackSegmentationDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_samples(self, path_to_samples, sample_kind):
        samples = [i for i in sorted(os.listdir(path_to_samples)) if 'GT' not in i]
        samples_read = list()

        for sample in samples:
            part = sample.split(".")[0]
            
            image_path = path_to_samples + "/" + sample
            seg_mask_path = path_to_samples + "/" + part + "_GT.jpg"
            
            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            image = self.to_tensor(image)

            seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)

            if sample_kind == 'pos':
                seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
            else:
                seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask)))
            
            seg_mask = self.to_tensor(self.downsize(seg_mask))
                
            samples_read.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, part))

        return samples_read

    def read_contents(self):
        #eager loading

        pos_samples, neg_samples = [], []

        path_to_positive_test_samples = "./datasets/crack_segmentation/test_positive"
        path_to_negative_test_samples = "./datasets/crack_segmentation/test_negative"

        path_to_positive_train_samples = "./datasets/crack_segmentation/train_positive"
        path_to_negative_train_samples = "./datasets/crack_segmentation/train_negative"

        if self.kind == 'TEST':
            # Test Positive
            self.pos_samples = self.read_samples(path_to_positive_test_samples, 'pos')
            # Test Negative
            self.neg_samples = self.read_samples(path_to_negative_test_samples, 'neg')
        
        elif self.kind == 'TRAIN':
            # Train Positive
            self.pos_samples = self.read_samples(path_to_positive_train_samples, 'pos')
            # Train Negative
            self.neg_samples = self.read_samples(path_to_negative_train_samples, 'neg')

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)

        self.len = len(pos_samples) + len(neg_samples)
        
        print(f"{self.kind}: Number of positives: {self.num_pos}, Number of negatives: {self.num_neg}, Sum: {self.len}")

        self.init_extra()