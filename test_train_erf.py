#imports
import os
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from erfnet_model import get_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#setup directory
MODEL_DIR = "/Users/main/grad_school/term2_assessments/GeoAI/models"
IMAGE_DIR = '/Users/main/grad_school/term2_assessments/GeoAI/preprocessed_data/images_rgb_aug'
MASK_DIR = '/Users/main/grad_school/term2_assessments/GeoAI/preprocessed_data/masks_rgb_aug'
os.makedirs(MODEL_DIR, exist_ok=True)

#model hyperparams
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_FOLDS = 2
LR = 2e-4  #learn rate
USE_FRACTION = 1.0 #adjust for testing only
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

#added this last minute to vizualize the models efficacy (or lack thereof)
def visualize_predictions(model, dataloader, num_samples=6, device=DEVICE):
    model.eval()
    
    # Get a batch of samples
    images, masks = next(iter(dataloader))
    images = images.to(device)
    masks = masks.to(device)
    
    # Generate predictions
    with torch.no_grad():
        outputs = model(images)
        # Resize outputs if needed
        if outputs.shape != masks.shape:
            outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
        predictions = (torch.sigmoid(outputs) > 0.5).float()
    
    # Move tensors to CPU for visualization
    images = images.cpu()
    masks = masks.cpu()
    predictions = predictions.cpu()
    
    # Visualize the samples
    num_samples = min(num_samples, len(images))
    plt.figure(figsize=(15, 3 * num_samples))
    
    for i in range(num_samples):
        # Original image (taking first 3 channels if more are available)
        img = images[i+3]
        if img.shape[0] > 3:
            # If more than 3 channels, either take RGB channels or create a composite
            # Here we'll take first 3 channels for simplicity
            img_display = img[:3]
        else:
            img_display = img
            
        # Normalize for display if needed
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(img_display.permute(1, 2, 0))  # CHW -> HWC for matplotlib
        plt.axis('off')
        
        # Ground truth mask
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(masks[i+3].squeeze(), cmap='binary')
        plt.axis('off')
        
        # Model prediction
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(predictions[i+3].squeeze(), cmap='binary')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("model_predictions.png")
    plt.show()
    print(f"Visualization saved to 'model_predictions.png'")

#filter out all 0 masks to help with imbalance (massive overfitting issues due to a majority all 0 images)
def filter_filenames_with_non_empty_masks(image_dir, mask_dir, filenames):
    valid_filenames = []

    for fname in tqdm(filenames, desc="Filtering masks"):
        mask_path = os.path.join(mask_dir, fname.replace("img_", "mask_"))
        if not os.path.exists(mask_path):
            continue
        mask = np.load(mask_path)
        if np.any(mask > 0):
            valid_filenames.append(fname)
    return valid_filenames


#dice Loss for imbalanced segmentation
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

import torch.nn.functional as F

#because of the augmented data this class handles much of the mismatch sizing and rotations
#this also helped with testing other models, UNET was too slow and not classifiction
#admittably there is a bit of bloat to the code used, I was unsure most of the time so fixes were additive not reductive
class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, filenames, target_size=(64, 64)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = filenames
        self.target_size = target_size  

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        mask_filename = image_filename.replace("img_", "mask_")

        img_path = os.path.join(self.image_dir, image_filename)
        mask_path = os.path.join(self.mask_dir, mask_filename)

        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        #normalize mask if it's in 0–255 format
        #this was in response to some weird error with the masks
        if mask.max() > 1:
            mask = mask / 255.0

        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask).unsqueeze(0)

        #resize for aug data
        _, h, w = image.shape
        if (h, w) != self.target_size:
            image = F.interpolate(image.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0), size=self.target_size, mode='nearest').squeeze(0)

        return image, mask


def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            # Resize outputs to match masks if needed
            if outputs.shape != masks.shape:
                outputs = F.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
            
            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(preds.cpu().numpy().reshape(-1))
            all_targets.extend(masks.cpu().numpy().reshape(-1))

    if np.sum(all_preds) == 0:
        print("all background BAD")

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    return acc, prec, rec, f1

def train_k_fold():
    all_filenames = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.npy') and f.startswith('img_')])
    all_filenames = all_filenames[:int(len(all_filenames) * USE_FRACTION)]
    all_filenames = filter_filenames_with_non_empty_masks(IMAGE_DIR, MASK_DIR, all_filenames)

    print(f"Using {len(all_filenames)} image-mask pairs with non-empty masks.")

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_filenames)):
        print(f"\n====== Fold {fold+1}/{NUM_FOLDS} ======")
        train_files = [all_filenames[i] for i in train_idx]
        val_files = [all_filenames[i] for i in val_idx][:10]  #limit for testing

        train_dataset = RoadDataset(IMAGE_DIR, MASK_DIR, train_files)
        val_dataset = RoadDataset(IMAGE_DIR, MASK_DIR, val_files)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        model = get_model(in_channels=4, num_classes=1).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        pos_weight = torch.tensor([9.0]).to(DEVICE)  #adjust this value based on your road-to-background ratio
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  #pass class weights to the BCE loss
        dice_loss = DiceLoss()

        for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

            for images, masks in progress_bar:
                images, masks = images.to(DEVICE), masks.to(DEVICE)

                #make masks have the correct shape: [batch_size, 1, H, W]
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                outputs = model(images)
                outputs = F.interpolate(outputs, size=(64, 64), mode='bilinear', align_corners=False)

                #make sure outputs have the correct shape: [batch_size, 1, H, W]
                if outputs.shape[1] != 1: 
                    outputs = outputs[:, :1, :, :] 

                loss = 0.3 * bce_loss(outputs, masks) + 0.7 * dice_loss(outputs, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            print(f"epoch {epoch+1} completed — Avg Loss: {epoch_loss / len(train_loader):.4f}")

        acc, prec, rec, f1 = evaluate_model(model, val_loader)
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

        print(f"Fold {fold+1} — Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

        visualize_predictions(model, val_loader, num_samples=5)

        #save the model
        model_path = os.path.join(MODEL_DIR, f"model_fold{fold+1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"saved model {model_path}")

    #visuals
    # visualize_predictions(model, val_loader, num_samples=5)
    # Print the average evaluation scores across all folds
    print("\n📊 Final Evaluation Across All Folds:")
    print(f"🔍 Avg Accuracy : {np.mean(acc_list):.4f}")
    print(f"🔍 Avg Precision: {np.mean(prec_list):.4f}")
    print(f"🔍 Avg Recall   : {np.mean(rec_list):.4f}")
    print(f"🔍 Avg F1 Score : {np.mean(f1_list):.4f}")

if __name__ == "__main__":
    train_k_fold()