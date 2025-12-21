"""
Quick Training Script - Single Split (No K-Fold)
=================================================

A faster training option when you want to:
- Test configurations quickly
- Iterate on hyperparameters
- Limited computational resources

Uses same improvements as full pipeline:
- Attention U-Net with EfficientNet encoder
- Dice-BCE hybrid loss
- Advanced augmentations with elastic deformation
- Early stopping

For production/publication, use train_sota.py with K-Fold CV.
"""

import os

# Fix HuggingFace progress bar errors
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# ============================================================================
# CONFIGURATION
# ============================================================================

class QuickConfig:
    """Configuration for quick training."""
    
    # Data paths
    DATA_PATH = "siim-acr-pneumothorax"
    TRAIN_CSV = "stage_1_train_images.csv"
    IMAGES_DIR = "png_images"
    MASKS_DIR = "png_masks"
    OUTPUT_DIR = "quick_output"
    
    # Model
    ENCODER = "efficientnet-b4"
    ENCODER_WEIGHTS = "imagenet"
    
    # Training
    IMG_SIZE = 512
    BATCH_SIZE = 4        # Safe for 8GB VRAM (RTX 4060)
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    VAL_SPLIT = 0.2
    
    # Early stopping
    PATIENCE = 10
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    MIXED_PRECISION = True
    
    SEED = 42


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        # BCE
        bce_loss = self.bce(pred, target)
        
        # Dice
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = (2. * intersection + 1e-6) / (pred_sigmoid.sum() + target.sum() + 1e-6)
        dice_loss = 1 - dice
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# ============================================================================
# AUGMENTATION
# ============================================================================

def get_train_transform(img_size):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# ============================================================================
# DATASET
# ============================================================================

class PneumothoraxDataset(Dataset):
    def __init__(self, df, images_dir, masks_dir, transform=None, img_size=512):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['new_filename']
        
        img_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Binarize mask
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        mask = mask.permute(2, 0, 1)
        
        return image, mask


# ============================================================================
# METRICS
# ============================================================================

def dice_score(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    return (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)


def iou_score(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0
    running_dice = 0
    
    pbar = tqdm(loader, desc="Train")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        running_dice += dice_score(outputs, masks).item() * images.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(loader.dataset), running_dice / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    running_dice = 0
    running_iou = 0
    
    for images, masks in tqdm(loader, desc="Val"):
        images = images.to(device)
        masks = masks.to(device)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        running_loss += loss.item() * images.size(0)
        running_dice += dice_score(outputs, masks).item() * images.size(0)
        running_iou += iou_score(outputs, masks).item() * images.size(0)
    
    n = len(loader.dataset)
    return running_loss / n, running_dice / n, running_iou / n


def train(config):
    """Main training function."""
    seed_everything(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("QUICK TRAINING - PNEUMOTHORAX SEGMENTATION")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Encoder: {config.ENCODER}")
    print(f"Image Size: {config.IMG_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print("="*60)
    
    # Load data
    train_csv = os.path.join(config.DATA_PATH, config.TRAIN_CSV)
    df = pd.read_csv(train_csv)
    print(f"\nTotal samples: {len(df)}")
    
    # Get stratification labels
    masks_dir = os.path.join(config.DATA_PATH, config.MASKS_DIR)
    has_pneumo = []
    for idx in tqdm(range(len(df)), desc="Checking masks"):
        filename = df.iloc[idx]['new_filename']
        mask = cv2.imread(os.path.join(masks_dir, filename), cv2.IMREAD_GRAYSCALE)
        has_pneumo.append(1 if mask is not None and mask.max() > 0 else 0)
    
    # Stratified split
    train_df, val_df = train_test_split(
        df, test_size=config.VAL_SPLIT,
        stratify=has_pneumo, random_state=config.SEED
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Datasets and loaders
    images_dir = os.path.join(config.DATA_PATH, config.IMAGES_DIR)
    
    train_dataset = PneumothoraxDataset(
        train_df, images_dir, masks_dir,
        get_train_transform(config.IMG_SIZE), config.IMG_SIZE
    )
    val_dataset = PneumothoraxDataset(
        val_df, images_dir, masks_dir,
        get_val_transform(), config.IMG_SIZE
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    # Model
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse"
    ).to(config.DEVICE)
    
    # Training setup
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6
    )
    scaler = GradScaler()
    
    # Training history
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
    best_dice = 0
    best_iou = 0
    best_model_path = None
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS} (LR: {optimizer.param_groups[0]['lr']:.2e})")
        
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, scaler, config.DEVICE
        )
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, config.DEVICE
        )
        
        scheduler.step()
        
        # Log history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            patience_counter = 0
            # Create descriptive filename with metrics
            model_name = f"pneumo_dice{val_dice:.4f}_iou{val_iou:.4f}_ep{epoch+1}.pth"
            best_model_path = os.path.join(config.OUTPUT_DIR, model_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'encoder': config.ENCODER,
                'patch_size': config.IMG_SIZE
            }, best_model_path)
            print(f"→ New best model saved: {model_name}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs, history['train_dice'], 'b-', label='Train Dice')
    axes[1].plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    axes[1].plot(epochs, history['val_iou'], 'g--', label='Val IoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Metric Curves')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'training_curves.png'), dpi=150)
    plt.close()
    
    # Visualize predictions
    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
    model.eval()
    
    fig, axes = plt.subplots(5, 3, figsize=(12, 15))
    
    indices = np.random.choice(len(val_dataset), 5, replace=False)
    for i, idx in enumerate(indices):
        image, mask = val_dataset[idx]
        
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(config.DEVICE))
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        
        # Denormalize image
        img_vis = image.permute(1, 2, 0).numpy()
        img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_vis = np.clip(img_vis, 0, 1)
        
        pred_binary = (pred > 0.5).astype(float)
        mask_np = mask.squeeze().numpy()
        
        # Dice for this sample
        intersection = (pred_binary * mask_np).sum()
        sample_dice = 2 * intersection / (pred_binary.sum() + mask_np.sum() + 1e-6)
        
        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title('Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_binary, cmap='gray')
        axes[i, 2].set_title(f'Prediction (Dice: {sample_dice:.3f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, 'predictions.png'), dpi=150)
    plt.close()
    
    # Save summary
    summary = {
        'best_dice': float(best_dice),
        'best_iou': float(max(history['val_iou'])),
        'final_train_loss': float(history['train_loss'][-1]),
        'final_val_loss': float(history['val_loss'][-1]),
        'epochs_trained': len(history['train_loss']),
        'config': {
            'encoder': config.ENCODER,
            'img_size': config.IMG_SIZE,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE
        }
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Best Validation IoU:  {best_iou:.4f}")
    if best_model_path:
        print(f"Model saved to: {best_model_path}")
    print("="*60)
    
    return history, best_dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='siim-acr-pneumothorax')
    parser.add_argument('--output-dir', type=str, default='quick_output')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--encoder', type=str, default='efficientnet-b4')
    
    args = parser.parse_args()
    
    config = QuickConfig()
    config.DATA_PATH = args.data_path
    config.OUTPUT_DIR = args.output_dir
    config.IMG_SIZE = args.img_size
    config.BATCH_SIZE = args.batch_size
    config.EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.ENCODER = args.encoder
    
    train(config)

