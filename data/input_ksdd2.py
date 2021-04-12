import numpy as np
import pickle
import os
from data.dataset import Dataset
from config import Config


def read_split(num_segmented: int, kind: str):
    fn = f"KSDD2/split_{num_segmented}.pyb"
    with open(f"splits/{fn}", "rb") as f:
        train_samples, test_samples = pickle.load(f)
        if kind == 'TRAIN':
            return train_samples
        elif kind == 'TEST':
            return test_samples
        else:
            raise Exception('Unknown')


class KSDD2Dataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(KSDD2Dataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_contents(self):
        pos_samples, neg_samples = [], []

        data_points = read_split(self.cfg.NUM_SEGMENTED, self.kind)

        for part, is_segmented in data_points:
            image_path = os.path.join(self.path, self.kind.lower(), f"{part}.png")
            seg_mask_path = os.path.join(self.path, self.kind.lower(), f"{part}_GT.png")

            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)

            if positive:
                image = self.to_tensor(image)
                seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                pos_samples.append((image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, part))
            else:
                image = self.to_tensor(image)
                seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask)))
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, part))

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = 2 * len(pos_samples) if self.kind in ['TRAIN'] else len(pos_samples) + len(neg_samples)

        self.init_extra()
