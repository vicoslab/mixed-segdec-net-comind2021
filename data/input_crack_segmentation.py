import os
from data.dataset import Dataset
from config import Config

class CrackSegmentationDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(CrackSegmentationDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_contents(self):
        #eager loading

        pos_samples, neg_samples = [], []

        path_to_positives = "./datasets/crack_segmentation/positive"
        path_to_negatives = "./datasets/crack_segmentation/negative"

        positives = os.listdir(path_to_positives)
        negatives = os.listdir(path_to_negatives)

        # Positives
        for i in range(0, len(positives), 2):
            image_path = path_to_positives + f"/{positives[i]}"
            seg_mask_path = path_to_positives + f"/{positives[i + 1]}"
            
            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            image = self.to_tensor(image)

            seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)

            seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
            seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))

            seg_mask = self.to_tensor(self.downsize(seg_mask))
            
            is_segmented = True
            part = positives[i].split(".")[0]

            pos_samples.append((image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, part))
        
        # Negatives
        for i in range(0, len(negatives), 2):
            image_path = path_to_negatives + f"/{negatives[i]}"
            seg_mask_path = path_to_negatives + f"/{negatives[i + 1]}"
            
            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            image = self.to_tensor(image)

            seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)

            seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
            seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))

            seg_mask = self.to_tensor(self.downsize(seg_mask))
            
            is_segmented = True
            part = negatives[i].split(".")[0]

            neg_samples.append((image, seg_mask, seg_loss_mask, is_segmented, image_path, seg_mask_path, part))
        
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)

        self.init_extra()