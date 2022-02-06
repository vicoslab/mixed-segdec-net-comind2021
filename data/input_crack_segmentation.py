import os
import numpy as np
from data.dataset import Dataset
from config import Config
from datetime import datetime

class CrackSegmentationDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(CrackSegmentationDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_samples(self, path_to_samples, sample_kind):
        samples = [i for i in sorted(os.listdir(path_to_samples)) if 'GT' not in i]

        for sample in samples:
            part = sample.split(".")[0]
            
            image_path = path_to_samples + "/" + sample
            seg_mask_path = path_to_samples + "/" + part + "_GT.jpg"
            
            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            image = self.to_tensor(image)

            seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)

            if sample_kind == 'pos':
                seg_loss_mask_original = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                seg_loss_mask_original = self.to_tensor(seg_loss_mask_original)
                seg_mask_original = self.to_tensor(seg_mask)
                self.pos_samples.append((image, True, image_path, seg_mask_path, part, seg_mask_original, seg_loss_mask_original))
            else:
                seg_loss_mask_original = self.to_tensor(np.ones_like(seg_mask))
                seg_mask_original = self.to_tensor(seg_mask)
                self.neg_samples.append((image, True, image_path, seg_mask_path, part, seg_mask_original, seg_loss_mask_original))

    def read_contents(self):
        #eager loading

        self.pos_samples = list()
        self.neg_samples = list()

        path_to_positive_test_samples = os.path.join(self.cfg.DATASET_PATH, 'test_positive')
        path_to_negative_test_samples = os.path.join(self.cfg.DATASET_PATH, 'test_negative')

        path_to_positive_train_samples = os.path.join(self.cfg.DATASET_PATH, 'train_positive')
        path_to_negative_train_samples = os.path.join(self.cfg.DATASET_PATH, 'train_negative')

        path_to_positive_val_samples = os.path.join(self.cfg.DATASET_PATH, 'val_positive')
        path_to_negative_val_samples = os.path.join(self.cfg.DATASET_PATH, 'val_negative')

        if self.kind == 'TEST':
            # Test Positive
            self.read_samples(path_to_positive_test_samples, 'pos')
            # Test Negative
            self.read_samples(path_to_negative_test_samples, 'neg')
        
        elif self.kind == 'TRAIN':
            # Train Positive
            self.read_samples(path_to_positive_train_samples, 'pos')
            # Train Negative
            self.read_samples(path_to_negative_train_samples, 'neg')

        elif self.kind == 'VAL':
            # Val Positive
            self.read_samples(path_to_positive_val_samples, 'pos')
            # Val negative
            self.read_samples(path_to_negative_val_samples, 'neg')

        self.num_pos = len(self.pos_samples)
        self.num_neg = len(self.neg_samples)

        self.len = self.num_pos + self.num_neg
        
        time = datetime.now().strftime("%d-%m-%y %H:%M")
        print(f"{time} {self.kind}: Number of positives: {self.num_pos}, Number of negatives: {self.num_neg}, Sum: {self.len}")

        self.init_extra()