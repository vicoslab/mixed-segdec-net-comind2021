import os
import numpy as np
from data.dataset import Dataset
from config import Config

class CrackSegmentationDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(CrackSegmentationDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_contents(self):
        #eager loading

        pos_samples, neg_samples = [], []

        path_to_positive_test_samples = "./datasets/crack_segmentation/test_positive"
        path_to_positive_train_samples = "./datasets/crack_segmentation/train_positive"

        path_to_negative_test_samples = "./datasets/crack_segmentation/test_negative"
        path_to_negative_train_samples = "./datasets/crack_segmentation/train_negative"

        if self.kind == 'TEST':
            # Test Positive
            positive_test_samples = os.listdir(path_to_positive_test_samples)
            for i in range(0, len(positive_test_samples), 2):
                image_path = path_to_positive_test_samples + "/" + positive_test_samples[i]
                seg_mask_path = path_to_positive_test_samples + "/" + positive_test_samples[i + 1]
                
                image = self.read_img_resize(image_path, self.grayscale, self.image_size)
                image = self.to_tensor(image)

                seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)

                seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
            
                seg_mask = self.to_tensor(self.downsize(seg_mask))

                is_segmented = True
                part = positive_test_samples[i].split(".")[0]
                
                pos_samples.append((image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, part))
            
            # Test Negative
            negative_test_samples = os.listdir(path_to_negative_test_samples)
            for i in range(0, len(negative_test_samples), 2):
                image_path = path_to_negative_test_samples + "/" + negative_test_samples[i]
                seg_mask_path = path_to_negative_test_samples + "/" + negative_test_samples[i + 1]
                
                image = self.read_img_resize(image_path, self.grayscale, self.image_size)
                image = self.to_tensor(image)

                seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)

                seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask)))
                seg_mask = self.to_tensor(self.downsize(seg_mask))

                part = negative_test_samples[i].split(".")[0]
                
                neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, part))
        
        elif self.kind == 'TRAIN':
            # Train Positive
            positive_train_samples = os.listdir(path_to_positive_train_samples)
            for i in range(0, len(positive_train_samples), 2):
                image_path = path_to_positive_train_samples + "/" + positive_train_samples[i]
                seg_mask_path = path_to_positive_train_samples + "/" + positive_train_samples[i + 1]
                
                image = self.read_img_resize(image_path, self.grayscale, self.image_size)
                image = self.to_tensor(image)

                seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)

                seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
            
                seg_mask = self.to_tensor(self.downsize(seg_mask))

                is_segmented = True
                part = positive_train_samples[i].split(".")[0]
                
                pos_samples.append((image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, part))
            
            # Train Negative
            negative_train_samples = os.listdir(path_to_negative_train_samples)
            for i in range(0, len(negative_train_samples), 2):
                image_path = path_to_negative_train_samples + "/" + negative_train_samples[i]
                seg_mask_path = path_to_negative_train_samples + "/" + negative_train_samples[i + 1]
                
                image = self.read_img_resize(image_path, self.grayscale, self.image_size)
                image = self.to_tensor(image)

                seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)

                seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask)))
                seg_mask = self.to_tensor(self.downsize(seg_mask))

                part = negative_train_samples[i].split(".")[0]
                
                neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, part))

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)

        self.len = len(pos_samples) + len(neg_samples)
        
        print(f"{self.kind}: Number of positives: {self.num_pos}, Number of negatives: {self.num_neg}, Sum: {self.len}")

        self.init_extra()        