import numpy as np
import os
import pickle
from data.dataset import Dataset


def read_split(num_segmented: int, fold: int, kind: str):
    fn = f"DAGM/split_{num_segmented}.pyb"
    with open(f"splits/{fn}", "rb") as f:
        train_samples, test_samples = pickle.load(f)
        if kind == 'TRAIN':
            return train_samples[fold - 1]
        elif kind == 'TEST':
            return test_samples[fold - 1]
        else:
            raise Exception('Unknown')


class DagmDataset(Dataset):
    def __init__(self, kind: str, cfg):
        super(DagmDataset, self).__init__(os.path.join(cfg.DATASET_PATH, f"Class{cfg.FOLD}"), cfg, kind)
        self.read_contents()

    def read_contents(self):
        pos_samples, neg_samples = [], []

        samples = read_split(self.cfg.NUM_SEGMENTED, self.cfg.FOLD, self.kind)

        sub_dir = self.kind.lower().capitalize()

        for image_name, is_segmented in samples:
            image_path = os.path.join(self.path, sub_dir, image_name)
            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            img_name_short = image_name[:-4]
            seg_mask_path = os.path.join(self.path, sub_dir, "Label",  f"{img_name_short}_label.PNG")

            if os.path.exists(seg_mask_path):
                seg_mask, _ = self.read_label_resize(seg_mask_path, self.image_size, dilate=self.cfg.DILATE)
                image = self.to_tensor(image)
                seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
                pos_samples.append((image, seg_mask, seg_loss_mask, is_segmented, image_path, None, img_name_short))

            else:
                seg_mask = np.zeros_like(image)
                image = self.to_tensor(image)
                seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask)))
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                neg_samples.append((image, seg_mask, seg_loss_mask, True, image_path, seg_mask_path, img_name_short))

        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = 2 * len(pos_samples) if self.kind in ['TRAIN'] else len(pos_samples) + len(neg_samples)

        self.init_extra()
