import os
import glob
import random
from PIL import Image
import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from unet import UNet  # assuming you're in the same dir as unet.py

# --- CONFIG ---
CONTROL_PATH = "/Users/daniel_kristensen/DAKI/DMIAI/2025/DM-i-AI-2025/tumor-segmentation/data/controls"
PATIENT_IMG_PATH = "/Users/daniel_kristensen/DAKI/DMIAI/2025/DM-i-AI-2025/tumor-segmentation/data/patients/imgs"
PATIENT_LABEL_PATH = "/Users/daniel_kristensen/DAKI/DMIAI/2025/DM-i-AI-2025/tumor-segmentation/data/patients/labels"

# Real run (slower)
#IMAGE_SIZE = (256, 256)
#BATCH_SIZE = 8
#EPOCHS = 30

# Test run (faster):
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 4
EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Dataset Class ---
class TumorSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("L")  # grayscale
        image = image.resize(IMAGE_SIZE)

        if self.mask_paths:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            mask = mask.resize(IMAGE_SIZE)
            mask = np.array(mask, dtype=np.float32) / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        else:
            mask = torch.zeros((1, *IMAGE_SIZE))

        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]

        return image, mask

# --- Load file paths ---
control_images = sorted(glob.glob(os.path.join(CONTROL_PATH, "*.png")))
patient_images = sorted(glob.glob(os.path.join(PATIENT_IMG_PATH, "*.png")))
patient_labels = sorted(glob.glob(os.path.join(PATIENT_LABEL_PATH, "*.png")))

all_images = control_images + patient_images
all_masks = [None] * len(control_images) + patient_labels

# --- Shuffle and split ---
data = list(zip(all_images, all_masks))
random.shuffle(data)
split1 = int(0.7 * len(data))
split2 = int(0.85 * len(data))
train_data = data[:split1]
val_data = data[split1:split2]
test_data = data[split2:]

train_dataset = TumorSegmentationDataset(
    [x[0] for x in train_data], [x[1] for x in train_data]
)
val_dataset = TumorSegmentationDataset(
    [x[0] for x in val_data], [x[1] for x in val_data]
)
test_dataset = TumorSegmentationDataset(
    [x[0] for x in test_data], [x[1] for x in test_data]
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# --- Dice + BCE Loss ---
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        smooth = 1.0

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 0.5 * self.bce(inputs, targets) + 0.5 * (1 - dice)

# --- Training ---
def train_model():
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = DiceBCELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "unet_tumor_segmentation.pt")
    print("Model saved as unet_tumor_segmentation.pt")

if __name__ == "__main__":
    train_model()
