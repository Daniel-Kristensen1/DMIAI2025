import os
import glob
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from unet import UNet  # Make sure unet.py is in the same folder or adjust import accordingly

# --- Config ---
CONTROL_PATH = "/Users/daniel_kristensen/DAKI/DMIAI/2025/DM-i-AI-2025/tumor-segmentation/data/controls"
PATIENT_IMG_PATH = "/Users/daniel_kristensen/DAKI/DMIAI/2025/DM-i-AI-2025/tumor-segmentation/data/patients/imgs"
PATIENT_LABEL_PATH = "/Users/daniel_kristensen/DAKI/DMIAI/2025/DM-i-AI-2025/tumor-segmentation/data/patients/labels"

IMAGE_SIZE = (128, 128)  # Same as training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Class ---
class TumorSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")
        image = image.resize(IMAGE_SIZE)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)

        if self.mask_paths and self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            mask = mask.resize(IMAGE_SIZE)
            mask = np.array(mask, dtype=np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)
        else:
            mask = torch.zeros((1, *IMAGE_SIZE), dtype=torch.float32)

        return image, mask

# --- Prepare test dataset ---
control_images = sorted(glob.glob(os.path.join(CONTROL_PATH, "*.png")))
patient_images = sorted(glob.glob(os.path.join(PATIENT_IMG_PATH, "*.png")))
patient_labels = sorted(glob.glob(os.path.join(PATIENT_LABEL_PATH, "*.png")))

all_images = control_images + patient_images
all_masks = [None] * len(control_images) + patient_labels

# Shuffle and split same as in training
data = list(zip(all_images, all_masks))
random.seed(42)  # fixed seed for reproducibility
random.shuffle(data)
split1 = int(0.7 * len(data))
split2 = int(0.85 * len(data))

test_data = data[split2:]
test_images = [x[0] for x in test_data]
test_masks = [x[1] for x in test_data]

test_dataset = TumorSegmentationDataset(test_images, test_masks)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# --- Load model ---
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load("unet_tumor_segmentation.pt", map_location=DEVICE))
model.eval()

# --- Dice score metric ---
def dice_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    smooth = 1.0

    intersection = (preds * targets).sum()
    dice = (2 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()

# --- Evaluation loop ---
dice_scores = []

with torch.no_grad():
    for imgs, masks in test_loader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        outputs = model(imgs)
        score = dice_score(outputs, masks)
        dice_scores.append(score)

avg_dice = sum(dice_scores) / len(dice_scores)
print(f"Average Dice Score on Test Set: {avg_dice:.4f}")
