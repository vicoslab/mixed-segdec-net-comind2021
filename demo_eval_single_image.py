from models import SegDecNet
import cv2
import torch
import numpy as np

INPUT_WIDTH = 512  # must be the same as it was during training
INPUT_HEIGHT = 1408  # must be the same as it was during training
INPUT_CHANNELS = 1  # must be the same as it was during training

device = "cpu"  # cpu or cuda:IX

model = SegDecNet(device, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)
model.set_gradient_multipliers(0)

model_path = "path_to_your_model"
model.load_state_dict(torch.load(model_path, map_location=device))

# %%
img_path = "path_to_the_test_image"
img = cv2.imread(img_path) if INPUT_CHANNELS == 3 else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
img = np.transpose(img, (2, 0, 1)) if INPUT_CHANNELS == 3 else img[np.newaxis]
img_t = torch.from_numpy(img)[np.newaxis].float() / 255.0  # must be [BATCH_SIZE x CHANNELS x HEIGHT x WIDTH]

dec_out, seg_out = model(img_t)
img_score = torch.sigmoid(dec_out)
print(img_score)
