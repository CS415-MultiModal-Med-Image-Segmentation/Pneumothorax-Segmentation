"""
State-of-the-Art Pneumothorax Segmentation Model
=================================================

Implementation based on expert research recommendations:
- Attention U-Net architecture with EfficientNet encoder
- Patch-based training from high-resolution images
- Dice-BCE hybrid loss for class imbalance
- K-Fold stratified cross-validation
- Advanced data augmentation with elastic deformations
- Sliding window inference with test-time augmentation

Target: Dice Coefficient > 0.85
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
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# ML/DL imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Segmentation models
import segmentation_models_pytorch as smp

# Augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Cross-validation
from sklearn.model_selection import StratifiedKFold, train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the training pipeline."""
    
    # Paths
    DATA_PATH = "siim-acr-pneumothorax"
    TRAIN_CSV = "stage_1_train_images.csv"
    TEST_CSV = "stage_1_test_images.csv"
    IMAGES_DIR = "png_images"
    MASKS_DIR = "png_masks"
    OUTPUT_DIR = "sota_output"
    
    # Image settings - HIGH RESOLUTION approach
    ORIGINAL_SIZE = 1024  # Original image size
    PATCH_SIZE = 512      # Training patch size (higher than 256 for detail)
    USE_PATCHES = True    # Enable patch-based training
    
    # Model settings
    ENCODER = "efficientnet-b4"  # Strong encoder for high-res
    ENCODER_WEIGHTS = "imagenet"
    ATTENTION_TYPE = "scse"  # Spatial and Channel Squeeze & Excitation
    
    # Training settings
    USE_KFOLD = False     # Set True for K-Fold CV (production), False for single split (faster)
    N_FOLDS = 5           # K-Fold cross-validation (only if USE_KFOLD=True)
    VAL_SPLIT = 0.2       # Validation split ratio (only if USE_KFOLD=False)
    BATCH_SIZE = 8        # Safe for 8GB VRAM (RTX 4060)
    EPOCHS = 50           # Max epochs
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    GRADIENT_ACCUMULATION = 1  # Effective batch size = 8 * 1 = 8
    
    # Early stopping
    PATIENCE = 15         # Early stopping patience
    MIN_DELTA = 1e-4      # Minimum improvement
    
    # Scheduler
    SCHEDULER = "cosine"  # cosine or reduce_on_plateau
    T_MAX = 50            # For cosine annealing
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True
    MIXED_PRECISION = True
    
    # Reproducibility
    SEED = 42
    
    # Inference
    SLIDING_WINDOW_OVERLAP = 0.5  # 50% overlap for sliding window
    TTA = True                     # Test-time augmentation


def seed_everything(seed: int):
    """Set seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice and BCE Loss.
    
    This hybrid approach:
    - BCE provides stable gradients early in training
    - Dice optimizes for region overlap (handles class imbalance)
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


class FocalDiceLoss(nn.Module):
    """
    Focal + Dice Loss for highly imbalanced segmentation.
    Focal loss down-weights easy examples and focuses on hard ones.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 dice_weight: float = 0.5, focal_weight: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Focal loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        focal = focal.mean()
        
        # Dice loss
        dice = self.dice_loss(pred, target)
        
        return self.focal_weight * focal + self.dice_weight * dice


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def get_training_augmentation(patch_size: int = 512) -> A.Compose:
    """
    Advanced augmentation pipeline for medical image segmentation.
    
    Key components:
    - Elastic deformation: Mimics physiological variations
    - Grid distortion: Additional non-rigid transformations
    - Affine: Standard geometric augmentations
    - Color/brightness: Handles imaging variations
    """
    return A.Compose([
        # Spatial augmentations
        A.HorizontalFlip(p=0.5),
        
        # Affine transformations
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
        
        # CRITICAL: Elastic deformation for medical imaging
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            p=0.3
        ),
        
        # Grid distortion for additional variation
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=0.3
        ),
        
        # Optical distortion
        A.OpticalDistortion(
            distort_limit=0.05,
            shift_limit=0.05,
            p=0.2
        ),
        
        # Intensity augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        
        # CLAHE for contrast enhancement (common in X-ray preprocessing)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        
        # Normalization and tensor conversion
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def get_validation_augmentation() -> A.Compose:
    """Validation augmentation - only normalization."""
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])


def get_tta_augmentations() -> List[A.Compose]:
    """Test-time augmentation transforms."""
    base_norm = A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
    
    return [
        # Original
        A.Compose([base_norm, ToTensorV2()]),
        # Horizontal flip
        A.Compose([A.HorizontalFlip(p=1.0), base_norm, ToTensorV2()]),
        # Slight rotation
        A.Compose([A.Rotate(limit=5, p=1.0), base_norm, ToTensorV2()]),
        # Brightness adjustment
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0), 
                   base_norm, ToTensorV2()]),
    ]


# ============================================================================
# DATASET
# ============================================================================

class PneumothoraxPatchDataset(Dataset):
    """
    Patch-based dataset for high-resolution pneumothorax segmentation.
    
    Features:
    - Loads full 1024x1024 images
    - Extracts random patches during training
    - Stratified by pneumothorax presence
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        masks_dir: str,
        transform: Optional[A.Compose] = None,
        patch_size: int = 512,
        is_training: bool = True,
        use_patches: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.patch_size = patch_size
        self.is_training = is_training
        self.use_patches = use_patches
        
        # Precompute which samples have pneumothorax for stratification
        self._compute_has_pneumothorax()
    
    def _compute_has_pneumothorax(self):
        """Check each mask for pneumothorax presence."""
        self.has_pneumothorax = []
        for idx in range(len(self.df)):
            filename = self.df.iloc[idx]['new_filename']
            mask_path = os.path.join(self.masks_dir, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                self.has_pneumothorax.append(1 if mask.max() > 0 else 0)
            else:
                self.has_pneumothorax.append(0)
        self.has_pneumothorax = np.array(self.has_pneumothorax)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _random_crop(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a random patch from image and mask."""
        h, w = image.shape[:2]
        
        # Ensure patch fits
        max_y = max(0, h - self.patch_size)
        max_x = max(0, w - self.patch_size)
        
        if max_y == 0 and max_x == 0:
            # Image is smaller than patch, resize
            image = cv2.resize(image, (self.patch_size, self.patch_size))
            mask = cv2.resize(mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
            return image, mask
        
        # Random crop position
        y = np.random.randint(0, max_y + 1)
        x = np.random.randint(0, max_x + 1)
        
        image_patch = image[y:y+self.patch_size, x:x+self.patch_size]
        mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
        
        return image_patch, mask_patch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.df.iloc[idx]['new_filename']
        
        img_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        
        # Load full resolution image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Convert to RGB (models expect 3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply patch extraction if enabled
        if self.use_patches and self.is_training:
            image, mask = self._random_crop(image, mask)
        else:
            # Resize to patch size for validation/inference
            image = cv2.resize(image, (self.patch_size, self.patch_size))
            mask = cv2.resize(mask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
        
        # Binarize mask
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure correct shape [C, H, W]
        mask = mask.permute(2, 0, 1) if mask.dim() == 3 else mask.unsqueeze(0)
        
        return image, mask


# ============================================================================
# MODEL
# ============================================================================

def get_model(config: Config) -> nn.Module:
    """
    Create Attention U-Net model with pretrained encoder.
    
    Using segmentation_models_pytorch with:
    - EfficientNet-B4 encoder for strong feature extraction
    - scse attention (Spatial and Channel Squeeze-Excitation)
    """
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None,  # Raw logits for BCEWithLogitsLoss
        decoder_attention_type=config.ATTENTION_TYPE
    )
    return model


def get_attention_unet_plus_plus(config: Config) -> nn.Module:
    """Alternative: U-Net++ with attention for deep supervision."""
    model = smp.UnetPlusPlus(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type=config.ATTENTION_TYPE
    )
    return model


# ============================================================================
# METRICS
# ============================================================================

def calculate_dice_coefficient(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """Calculate Dice coefficient."""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


def calculate_iou(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """Calculate Intersection over Union (Jaccard Index)."""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def calculate_sensitivity(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """Calculate Sensitivity (Recall) - critical for clinical safety."""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    
    sensitivity = (true_positive + smooth) / (actual_positive + smooth)
    return sensitivity.item()


# ============================================================================
# TRAINING
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: GradScaler,
    use_amp: bool = True,
    grad_accum_steps: int = 1
) -> Tuple[float, float]:
    """Train for one epoch with gradient accumulation."""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
        
        scaler.scale(loss).backward()
        
        # Step optimizer every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * grad_accum_steps * images.size(0)
        running_dice += calculate_dice_coefficient(outputs, masks) * images.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{calculate_dice_coefficient(outputs, masks):.4f}'
        })
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_dice = running_dice / len(loader.dataset)
    
    return epoch_loss, epoch_dice


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = True
) -> Tuple[float, float, float, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_sensitivity = 0.0
    
    pbar = tqdm(loader, desc="Validation")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        batch_dice = calculate_dice_coefficient(outputs, masks)
        batch_iou = calculate_iou(outputs, masks)
        batch_sens = calculate_sensitivity(outputs, masks)
        
        running_loss += loss.item() * images.size(0)
        running_dice += batch_dice * images.size(0)
        running_iou += batch_iou * images.size(0)
        running_sensitivity += batch_sens * images.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{batch_dice:.4f}'
        })
    
    n = len(loader.dataset)
    return running_loss/n, running_dice/n, running_iou/n, running_sensitivity/n


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Config
) -> Dict:
    """Train a single fold."""
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1}/{config.N_FOLDS}")
    print(f"{'='*60}")
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    # Create datasets
    train_dataset = PneumothoraxPatchDataset(
        df=train_df,
        images_dir=os.path.join(config.DATA_PATH, config.IMAGES_DIR),
        masks_dir=os.path.join(config.DATA_PATH, config.MASKS_DIR),
        transform=get_training_augmentation(config.PATCH_SIZE),
        patch_size=config.PATCH_SIZE,
        is_training=True,
        use_patches=config.USE_PATCHES
    )
    
    val_dataset = PneumothoraxPatchDataset(
        df=val_df,
        images_dir=os.path.join(config.DATA_PATH, config.IMAGES_DIR),
        masks_dir=os.path.join(config.DATA_PATH, config.MASKS_DIR),
        transform=get_validation_augmentation(),
        patch_size=config.PATCH_SIZE,
        is_training=False,
        use_patches=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Model
    model = get_model(config).to(config.DEVICE)
    
    # Loss, optimizer, scheduler
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    if config.SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.T_MAX, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )
    
    scaler = GradScaler(enabled=config.MIXED_PRECISION)
    early_stopping = EarlyStopping(patience=config.PATIENCE)
    
    # Training history
    history = {
        'train_loss': [], 'train_dice': [],
        'val_loss': [], 'val_dice': [], 'val_iou': [], 'val_sensitivity': []
    }
    best_dice = 0.0
    best_iou = 0.0
    best_model_path = None  # Will be set when saving
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            config.DEVICE, scaler, config.MIXED_PRECISION,
            grad_accum_steps=config.GRADIENT_ACCUMULATION
        )
        
        # Validate
        val_loss, val_dice, val_iou, val_sensitivity = validate(
            model, val_loader, criterion, config.DEVICE, config.MIXED_PRECISION
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        history['val_sensitivity'].append(val_sensitivity)
        
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | "
              f"Val IoU: {val_iou:.4f} | Val Sensitivity: {val_sensitivity:.4f}")
        
        # Scheduler step
        if config.SCHEDULER == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_dice)
        
        # Save best model with metrics in filename
        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            # Create descriptive filename: pneumo_fold{N}_dice{D}_iou{I}_ep{E}.pth
            model_name = f"pneumo_fold{fold}_dice{val_dice:.4f}_iou{val_iou:.4f}_ep{epoch+1}.pth"
            best_model_path = os.path.join(config.OUTPUT_DIR, model_name)
            torch.save({
                'epoch': epoch,
                'fold': fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'val_sensitivity': val_sensitivity,
                'encoder': config.ENCODER,
                'patch_size': config.PATCH_SIZE,
                'config': config.__dict__
            }, best_model_path)
            print(f"→ New best model saved: {model_name}")
        
        # Early stopping
        if early_stopping(val_dice):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    # Load best model for final metrics
    if best_model_path and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return {
        'fold': fold,
        'best_dice': best_dice,
        'best_iou': best_iou,
        'history': history,
        'model_path': best_model_path
    }


# ============================================================================
# SLIDING WINDOW INFERENCE
# ============================================================================

@torch.no_grad()
def sliding_window_inference(
    model: nn.Module,
    image: np.ndarray,
    patch_size: int = 512,
    overlap: float = 0.5,
    device: str = 'cuda',
    transform: Optional[A.Compose] = None
) -> np.ndarray:
    """
    Perform sliding window inference on a high-resolution image.
    
    Args:
        model: Trained model
        image: Original high-resolution image (H, W, C)
        patch_size: Size of patches for inference
        overlap: Overlap ratio between patches
        device: Computing device
        transform: Normalization transform
    
    Returns:
        Full resolution prediction mask
    """
    model.eval()
    
    h, w = image.shape[:2]
    stride = int(patch_size * (1 - overlap))
    
    # Create output and weight accumulators
    pred_mask = np.zeros((h, w), dtype=np.float32)
    weight_mask = np.zeros((h, w), dtype=np.float32)
    
    # Gaussian weight for smooth blending
    gaussian_weight = _create_gaussian_weight(patch_size)
    
    # Slide over image
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Extract patch
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Apply transform
            if transform:
                transformed = transform(image=patch)
                patch_tensor = transformed['image'].unsqueeze(0).to(device)
            else:
                patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            # Predict
            with autocast():
                pred = model(patch_tensor)
                pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            
            # Accumulate with Gaussian weighting
            pred_mask[y:y+patch_size, x:x+patch_size] += pred * gaussian_weight
            weight_mask[y:y+patch_size, x:x+patch_size] += gaussian_weight
    
    # Handle edges if image size doesn't perfectly divide
    # Right edge
    if (w - patch_size) % stride != 0:
        x = w - patch_size
        for y in range(0, h - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            if transform:
                transformed = transform(image=patch)
                patch_tensor = transformed['image'].unsqueeze(0).to(device)
            else:
                patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            with autocast():
                pred = model(patch_tensor)
                pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            
            pred_mask[y:y+patch_size, x:x+patch_size] += pred * gaussian_weight
            weight_mask[y:y+patch_size, x:x+patch_size] += gaussian_weight
    
    # Bottom edge
    if (h - patch_size) % stride != 0:
        y = h - patch_size
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            if transform:
                transformed = transform(image=patch)
                patch_tensor = transformed['image'].unsqueeze(0).to(device)
            else:
                patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            with autocast():
                pred = model(patch_tensor)
                pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            
            pred_mask[y:y+patch_size, x:x+patch_size] += pred * gaussian_weight
            weight_mask[y:y+patch_size, x:x+patch_size] += gaussian_weight
    
    # Bottom-right corner
    if (h - patch_size) % stride != 0 and (w - patch_size) % stride != 0:
        y = h - patch_size
        x = w - patch_size
        patch = image[y:y+patch_size, x:x+patch_size]
        if transform:
            transformed = transform(image=patch)
            patch_tensor = transformed['image'].unsqueeze(0).to(device)
        else:
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        with autocast():
            pred = model(patch_tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        
        pred_mask[y:y+patch_size, x:x+patch_size] += pred * gaussian_weight
        weight_mask[y:y+patch_size, x:x+patch_size] += gaussian_weight
    
    # Normalize by weights
    weight_mask[weight_mask == 0] = 1  # Avoid division by zero
    pred_mask /= weight_mask
    
    return pred_mask


def _create_gaussian_weight(size: int, sigma: Optional[float] = None) -> np.ndarray:
    """Create 2D Gaussian weight matrix for smooth blending."""
    if sigma is None:
        sigma = size / 4
    
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    xx, yy = np.meshgrid(x, y)
    
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return gaussian


@torch.no_grad()
def inference_with_tta(
    model: nn.Module,
    image: np.ndarray,
    device: str = 'cuda',
    patch_size: int = 512
) -> np.ndarray:
    """
    Test-time augmentation inference.
    Averages predictions from multiple augmented versions.
    """
    model.eval()
    
    tta_transforms = get_tta_augmentations()
    predictions = []
    
    for transform in tta_transforms:
        pred = sliding_window_inference(
            model, image, patch_size=patch_size,
            device=device, transform=transform
        )
        predictions.append(pred)
    
    # Average predictions
    final_pred = np.mean(predictions, axis=0)
    return final_pred


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(history: Dict, fold: int, output_dir: str):
    """Plot training curves for a fold."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice
    axes[0, 1].plot(epochs, history['train_dice'], 'b-', label='Train Dice')
    axes[0, 1].plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('Dice Score Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU
    axes[1, 0].plot(epochs, history['val_iou'], 'g-', label='Val IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('Validation IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Sensitivity
    axes[1, 1].plot(epochs, history['val_sensitivity'], 'm-', label='Val Sensitivity')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Sensitivity')
    axes[1, 1].set_title('Validation Sensitivity (Recall)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves_fold{fold}.png'), dpi=150)
    plt.close()


def plot_cv_summary(all_results: List[Dict], output_dir: str):
    """Plot cross-validation summary."""
    dices = [r['best_dice'] for r in all_results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    folds = [f"Fold {i+1}" for i in range(len(dices))]
    bars = ax.bar(folds, dices, color='steelblue', edgecolor='navy')
    
    # Add mean line
    mean_dice = np.mean(dices)
    ax.axhline(y=mean_dice, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_dice:.4f}')
    
    # Add std band
    std_dice = np.std(dices)
    ax.fill_between(
        range(-1, len(dices) + 1),
        mean_dice - std_dice,
        mean_dice + std_dice,
        alpha=0.2, color='red',
        label=f'Std: ±{std_dice:.4f}'
    )
    
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title(f'K-Fold Cross-Validation Results\nMean Dice: {mean_dice:.4f} ± {std_dice:.4f}', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, dice in zip(bars, dices):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{dice:.4f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_summary.png'), dpi=150)
    plt.close()


def visualize_predictions(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    output_dir: str,
    num_samples: int = 8
):
    """Visualize model predictions on validation samples."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3))
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device))
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        
        # Denormalize image for visualization
        img_vis = image.permute(1, 2, 0).numpy()
        img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_vis = np.clip(img_vis, 0, 1)
        
        mask_np = mask.squeeze().numpy()
        pred_binary = (pred > 0.5).astype(np.float32)
        
        # Dice for this sample
        intersection = (pred_binary * mask_np).sum()
        dice = 2 * intersection / (pred_binary.sum() + mask_np.sum() + 1e-6)
        
        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_binary, cmap='gray')
        axes[i, 2].set_title(f'Prediction (Dice: {dice:.3f})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'), dpi=150)
    plt.close()


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline - supports both single split and K-Fold CV."""
    config = Config()
    seed_everything(config.SEED)
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("PNEUMOTHORAX SEGMENTATION - STATE-OF-THE-ART TRAINING")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Encoder: {config.ENCODER}")
    print(f"Patch Size: {config.PATCH_SIZE}")
    print(f"Batch Size: {config.BATCH_SIZE} (effective: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION})")
    print(f"Mode: {'K-Fold CV (K=' + str(config.N_FOLDS) + ')' if config.USE_KFOLD else 'Single Split'}")
    print(f"Max Epochs: {config.EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Early Stopping Patience: {config.PATIENCE}")
    print("="*60)
    
    # Load data
    train_csv_path = os.path.join(config.DATA_PATH, config.TRAIN_CSV)
    df = pd.read_csv(train_csv_path)
    print(f"\nTotal training samples: {len(df)}")
    
    # Get stratification labels (presence of pneumothorax)
    images_dir = os.path.join(config.DATA_PATH, config.IMAGES_DIR)
    masks_dir = os.path.join(config.DATA_PATH, config.MASKS_DIR)
    
    print("Computing stratification labels...")
    has_pneumothorax = []
    for idx in tqdm(range(len(df))):
        filename = df.iloc[idx]['new_filename']
        mask_path = os.path.join(masks_dir, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        has_pneumothorax.append(1 if mask is not None and mask.max() > 0 else 0)
    
    stratify_labels = np.array(has_pneumothorax)
    print(f"Samples with pneumothorax: {stratify_labels.sum()} ({100*stratify_labels.mean():.1f}%)")
    print(f"Samples without pneumothorax: {len(stratify_labels) - stratify_labels.sum()}")
    
    all_results = []
    
    if config.USE_KFOLD:
        # ========== K-FOLD CROSS-VALIDATION ==========
        print(f"\n>>> Running {config.N_FOLDS}-Fold Cross-Validation...")
        kfold = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(df, stratify_labels)):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            fold_result = train_fold(fold, train_df, val_df, config)
            all_results.append(fold_result)
            
            # Plot training curves for this fold
            plot_training_curves(fold_result['history'], fold, config.OUTPUT_DIR)
        
        # Summary statistics
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS")
        print("="*60)
        
        dices = [r['best_dice'] for r in all_results]
        mean_dice = np.mean(dices)
        std_dice = np.std(dices)
        
        for i, r in enumerate(all_results):
            print(f"Fold {i+1}: Dice = {r['best_dice']:.4f}")
        
        print("-"*40)
        print(f"Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"Max Dice:  {max(dices):.4f}")
        print(f"Min Dice:  {min(dices):.4f}")
        
        # Plot CV summary
        plot_cv_summary(all_results, config.OUTPUT_DIR)
        
        best_fold = np.argmax(dices)
        print(f"\nBest fold: {best_fold + 1} with Dice = {dices[best_fold]:.4f}")
        
        # Save summary
        summary = {
            'mode': 'kfold',
            'mean_dice': float(mean_dice),
            'std_dice': float(std_dice),
            'fold_dices': [float(d) for d in dices],
            'best_fold': int(best_fold),
            'config': {k: str(v) for k, v in config.__dict__.items() if not k.startswith('_')}
        }
    
    else:
        # ========== SINGLE SPLIT TRAINING ==========
        print(f"\n>>> Running Single Split Training (val_split={config.VAL_SPLIT})...")
        
        train_df, val_df = train_test_split(
            df, test_size=config.VAL_SPLIT,
            stratify=stratify_labels, random_state=config.SEED
        )
        
        fold_result = train_fold(0, train_df, val_df, config)
        all_results.append(fold_result)
        
        # Plot training curves
        plot_training_curves(fold_result['history'], 0, config.OUTPUT_DIR)
        
        # Summary
        print("\n" + "="*60)
        print("TRAINING RESULTS")
        print("="*60)
        print(f"Best Validation Dice: {fold_result['best_dice']:.4f}")
        
        summary = {
            'mode': 'single_split',
            'best_dice': float(fold_result['best_dice']),
            'val_split': config.VAL_SPLIT,
            'config': {k: str(v) for k, v in config.__dict__.items() if not k.startswith('_')}
        }
    
    # Save summary
    import json
    with open(os.path.join(config.OUTPUT_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {config.OUTPUT_DIR}/")
    print("Training complete!")
    
    return all_results, config


if __name__ == "__main__":
    results, config = main()

