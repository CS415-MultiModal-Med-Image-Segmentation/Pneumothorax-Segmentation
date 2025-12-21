"""
Evaluation and Inference Script for Pneumothorax Segmentation
==============================================================

This script provides:
1. Model evaluation on test set
2. Sliding window inference for high-resolution images
3. Test-time augmentation (TTA)
4. Visualization of predictions
5. Comprehensive metrics calculation
"""

import os

# Fix HuggingFace progress bar errors
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import cv2
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple, Optional, List
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================================
# CONFIGURATION (should match training)
# ============================================================================

class EvalConfig:
    DATA_PATH = "siim-acr-pneumothorax"
    TEST_CSV = "stage_1_test_images.csv"
    IMAGES_DIR = "png_images"
    MASKS_DIR = "png_masks"
    OUTPUT_DIR = "sota_output"
    
    ENCODER = "efficientnet-b4"
    PATCH_SIZE = 512
    BATCH_SIZE = 4
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    
    # Inference settings
    SLIDING_WINDOW_OVERLAP = 0.5
    TTA_ENABLED = True
    THRESHOLD = 0.5


# ============================================================================
# DATASET
# ============================================================================

class TestDataset(Dataset):
    """Test dataset for evaluation."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        masks_dir: str,
        img_size: int = 512,
        transform: Optional[A.Compose] = None
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['new_filename']
        
        img_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Store original size for reference
        orig_h, orig_w = image.shape
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        if mask is not None:
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        mask = np.expand_dims(mask, axis=-1)
        
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        mask = mask.permute(2, 0, 1) if mask.dim() == 3 else mask.unsqueeze(0)
        
        return image, mask, filename, (orig_h, orig_w)


class FullResDataset(Dataset):
    """Full resolution dataset for sliding window inference."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        masks_dir: str
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['new_filename']
        
        img_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if mask is not None:
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        
        return image, mask, filename


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(checkpoint_path: str, config: EvalConfig) -> nn.Module:
    """Load trained model from checkpoint."""
    model = smp.Unet(
        encoder_name=config.ENCODER,
        encoder_weights=None,  # Don't load ImageNet weights, we have trained weights
        in_channels=3,
        classes=1,
        activation=None,
        decoder_attention_type="scse"
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Checkpoint validation Dice: {checkpoint.get('val_dice', 'N/A'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(config.DEVICE)
    model.eval()
    
    return model


def load_ensemble(model_paths: List[str], config: EvalConfig) -> List[nn.Module]:
    """Load multiple models for ensemble prediction."""
    models = []
    for path in model_paths:
        model = load_model(path, config)
        models.append(model)
    return models


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict:
    """Calculate comprehensive segmentation metrics."""
    pred_binary = (pred > threshold).astype(np.float32)
    
    smooth = 1e-6
    
    # True/False Positives/Negatives
    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    fn = ((1 - pred_binary) * target).sum()
    tn = ((1 - pred_binary) * (1 - target)).sum()
    
    # Dice (F1)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    
    # IoU (Jaccard)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    
    # Precision
    precision = (tp + smooth) / (tp + fp + smooth)
    
    # Recall (Sensitivity)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    # Specificity
    specificity = (tn + smooth) / (tn + fp + smooth)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


# ============================================================================
# INFERENCE
# ============================================================================

def get_transform():
    """Get normalization transform."""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def _create_gaussian_weight(size: int, sigma: Optional[float] = None) -> np.ndarray:
    """Create 2D Gaussian weight for smooth blending."""
    if sigma is None:
        sigma = size / 4
    x = np.arange(size) - size // 2
    y = np.arange(size) - size // 2
    xx, yy = np.meshgrid(x, y)
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return gaussian


@torch.no_grad()
def sliding_window_inference(
    model: nn.Module,
    image: np.ndarray,
    patch_size: int = 512,
    overlap: float = 0.5,
    device: str = 'cuda'
) -> np.ndarray:
    """Sliding window inference with Gaussian blending."""
    model.eval()
    transform = get_transform()
    
    h, w = image.shape[:2]
    stride = int(patch_size * (1 - overlap))
    
    pred_mask = np.zeros((h, w), dtype=np.float32)
    weight_mask = np.zeros((h, w), dtype=np.float32)
    gaussian_weight = _create_gaussian_weight(patch_size)
    
    # Pad image if necessary
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    
    if pad_h > 0 or pad_w > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        pred_mask = np.zeros((h + pad_h, w + pad_w), dtype=np.float32)
        weight_mask = np.zeros((h + pad_h, w + pad_w), dtype=np.float32)
    
    padded_h, padded_w = image.shape[:2]
    
    for y in range(0, padded_h - patch_size + 1, stride):
        for x in range(0, padded_w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            
            transformed = transform(image=patch)
            patch_tensor = transformed['image'].unsqueeze(0).to(device)
            
            with autocast():
                pred = model(patch_tensor)
                pred = torch.sigmoid(pred).squeeze().cpu().numpy()
            
            pred_mask[y:y+patch_size, x:x+patch_size] += pred * gaussian_weight
            weight_mask[y:y+patch_size, x:x+patch_size] += gaussian_weight
    
    # Handle remaining edges
    if (padded_w - patch_size) % stride != 0:
        x = padded_w - patch_size
        for y in range(0, padded_h - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            transformed = transform(image=patch)
            patch_tensor = transformed['image'].unsqueeze(0).to(device)
            with autocast():
                pred = torch.sigmoid(model(patch_tensor)).squeeze().cpu().numpy()
            pred_mask[y:y+patch_size, x:x+patch_size] += pred * gaussian_weight
            weight_mask[y:y+patch_size, x:x+patch_size] += gaussian_weight
    
    if (padded_h - patch_size) % stride != 0:
        y = padded_h - patch_size
        for x in range(0, padded_w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            transformed = transform(image=patch)
            patch_tensor = transformed['image'].unsqueeze(0).to(device)
            with autocast():
                pred = torch.sigmoid(model(patch_tensor)).squeeze().cpu().numpy()
            pred_mask[y:y+patch_size, x:x+patch_size] += pred * gaussian_weight
            weight_mask[y:y+patch_size, x:x+patch_size] += gaussian_weight
    
    weight_mask[weight_mask == 0] = 1
    pred_mask /= weight_mask
    
    # Remove padding
    pred_mask = pred_mask[:h, :w]
    
    return pred_mask


@torch.no_grad()
def inference_with_tta(
    model: nn.Module,
    image: np.ndarray,
    patch_size: int = 512,
    overlap: float = 0.5,
    device: str = 'cuda'
) -> np.ndarray:
    """Test-time augmentation with horizontal flip."""
    # Original prediction
    pred1 = sliding_window_inference(model, image, patch_size, overlap, device)
    
    # Horizontal flip
    image_flipped = np.fliplr(image).copy()
    pred2 = sliding_window_inference(model, image_flipped, patch_size, overlap, device)
    pred2 = np.fliplr(pred2)
    
    # Average predictions
    return (pred1 + pred2) / 2


@torch.no_grad()
def ensemble_inference(
    models: List[nn.Module],
    image: np.ndarray,
    patch_size: int = 512,
    overlap: float = 0.5,
    device: str = 'cuda',
    use_tta: bool = True
) -> np.ndarray:
    """Ensemble prediction from multiple models."""
    predictions = []
    
    for model in models:
        if use_tta:
            pred = inference_with_tta(model, image, patch_size, overlap, device)
        else:
            pred = sliding_window_inference(model, image, patch_size, overlap, device)
        predictions.append(pred)
    
    return np.mean(predictions, axis=0)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_standard(
    model: nn.Module,
    test_loader: DataLoader,
    config: EvalConfig
) -> dict:
    """Standard evaluation on resized images."""
    model.eval()
    
    all_metrics = []
    
    pbar = tqdm(test_loader, desc="Evaluating")
    for images, masks, filenames, orig_sizes in pbar:
        images = images.to(config.DEVICE)
        
        with autocast():
            preds = model(images)
            preds = torch.sigmoid(preds)
        
        preds = preds.cpu().numpy()
        masks = masks.cpu().numpy()
        
        for i in range(len(preds)):
            metrics = calculate_metrics(preds[i].squeeze(), masks[i].squeeze(), config.THRESHOLD)
            metrics['filename'] = filenames[i]
            all_metrics.append(metrics)
    
    # Aggregate metrics
    dice_scores = [m['dice'] for m in all_metrics]
    iou_scores = [m['iou'] for m in all_metrics]
    recall_scores = [m['recall'] for m in all_metrics]
    precision_scores = [m['precision'] for m in all_metrics]
    
    return {
        'dice_mean': np.mean(dice_scores),
        'dice_std': np.std(dice_scores),
        'iou_mean': np.mean(iou_scores),
        'iou_std': np.std(iou_scores),
        'recall_mean': np.mean(recall_scores),
        'recall_std': np.std(recall_scores),
        'precision_mean': np.mean(precision_scores),
        'precision_std': np.std(precision_scores),
        'all_metrics': all_metrics
    }


def evaluate_sliding_window(
    model: nn.Module,
    dataset: FullResDataset,
    config: EvalConfig,
    use_tta: bool = True
) -> dict:
    """Evaluation using sliding window inference."""
    model.eval()
    
    all_metrics = []
    
    pbar = tqdm(range(len(dataset)), desc="Evaluating (Sliding Window)")
    for idx in pbar:
        image, mask, filename = dataset[idx]
        
        if use_tta:
            pred = inference_with_tta(
                model, image, config.PATCH_SIZE,
                config.SLIDING_WINDOW_OVERLAP, config.DEVICE
            )
        else:
            pred = sliding_window_inference(
                model, image, config.PATCH_SIZE,
                config.SLIDING_WINDOW_OVERLAP, config.DEVICE
            )
        
        metrics = calculate_metrics(pred, mask, config.THRESHOLD)
        metrics['filename'] = filename
        all_metrics.append(metrics)
        
        pbar.set_postfix({'dice': f"{metrics['dice']:.4f}"})
    
    # Aggregate
    dice_scores = [m['dice'] for m in all_metrics]
    iou_scores = [m['iou'] for m in all_metrics]
    recall_scores = [m['recall'] for m in all_metrics]
    
    return {
        'dice_mean': np.mean(dice_scores),
        'dice_std': np.std(dice_scores),
        'iou_mean': np.mean(iou_scores),
        'iou_std': np.std(iou_scores),
        'recall_mean': np.mean(recall_scores),
        'recall_std': np.std(recall_scores),
        'all_metrics': all_metrics
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(
    model: nn.Module,
    dataset: FullResDataset,
    config: EvalConfig,
    num_samples: int = 10,
    output_path: str = 'evaluation_results.png'
):
    """Visualize predictions vs ground truth."""
    model.eval()
    
    # Sample indices
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples * 3))
    
    for i, idx in enumerate(indices):
        image, mask, filename = dataset[idx]
        
        pred = inference_with_tta(
            model, image, config.PATCH_SIZE,
            config.SLIDING_WINDOW_OVERLAP, config.DEVICE
        )
        pred_binary = (pred > config.THRESHOLD).astype(np.float32)
        
        metrics = calculate_metrics(pred, mask, config.THRESHOLD)
        
        # Original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Input: {filename[:20]}...')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction probability
        axes[i, 2].imshow(pred, cmap='jet', vmin=0, vmax=1)
        axes[i, 2].set_title('Prediction (Prob)')
        axes[i, 2].axis('off')
        
        # Binary prediction
        axes[i, 3].imshow(pred_binary, cmap='gray')
        axes[i, 3].set_title(f'Binary (Dice: {metrics["dice"]:.3f})')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")


def plot_metrics_distribution(metrics: List[dict], output_path: str):
    """Plot distribution of metrics across test set."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    dice_scores = [m['dice'] for m in metrics]
    iou_scores = [m['iou'] for m in metrics]
    recall_scores = [m['recall'] for m in metrics]
    precision_scores = [m['precision'] for m in metrics]
    
    # Dice distribution
    axes[0, 0].hist(dice_scores, bins=50, color='steelblue', edgecolor='navy', alpha=0.7)
    axes[0, 0].axvline(np.mean(dice_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(dice_scores):.4f}')
    axes[0, 0].set_xlabel('Dice Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Dice Score Distribution')
    axes[0, 0].legend()
    
    # IoU distribution
    axes[0, 1].hist(iou_scores, bins=50, color='forestgreen', edgecolor='darkgreen', alpha=0.7)
    axes[0, 1].axvline(np.mean(iou_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(iou_scores):.4f}')
    axes[0, 1].set_xlabel('IoU Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('IoU Distribution')
    axes[0, 1].legend()
    
    # Recall distribution
    axes[1, 0].hist(recall_scores, bins=50, color='coral', edgecolor='darkred', alpha=0.7)
    axes[1, 0].axvline(np.mean(recall_scores), color='blue', linestyle='--',
                       label=f'Mean: {np.mean(recall_scores):.4f}')
    axes[1, 0].set_xlabel('Recall (Sensitivity)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Recall Distribution')
    axes[1, 0].legend()
    
    # Precision distribution
    axes[1, 1].hist(precision_scores, bins=50, color='orchid', edgecolor='purple', alpha=0.7)
    axes[1, 1].axvline(np.mean(precision_scores), color='blue', linestyle='--',
                       label=f'Mean: {np.mean(precision_scores):.4f}')
    axes[1, 1].set_xlabel('Precision')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Precision Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics distribution to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Pneumothorax Segmentation Model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, default='siim-acr-pneumothorax', help='Path to data')
    parser.add_argument('--output-dir', type=str, default='evaluation_output', help='Output directory')
    parser.add_argument('--sliding-window', action='store_true', help='Use sliding window inference')
    parser.add_argument('--tta', action='store_true', default=True, help='Use test-time augmentation')
    parser.add_argument('--num-vis', type=int, default=10, help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    config = EvalConfig()
    config.DATA_PATH = args.data_path
    config.OUTPUT_DIR = args.output_dir
    config.TTA_ENABLED = args.tta
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("PNEUMOTHORAX SEGMENTATION - EVALUATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Device: {config.DEVICE}")
    print(f"Sliding Window: {args.sliding_window}")
    print(f"TTA: {config.TTA_ENABLED}")
    print("="*60)
    
    # Load model
    model = load_model(args.model, config)
    
    # Load test data
    test_csv = os.path.join(config.DATA_PATH, config.TEST_CSV)
    test_df = pd.read_csv(test_csv)
    print(f"\nTest samples: {len(test_df)}")
    
    if args.sliding_window:
        # Full resolution evaluation
        dataset = FullResDataset(
            test_df,
            os.path.join(config.DATA_PATH, config.IMAGES_DIR),
            os.path.join(config.DATA_PATH, config.MASKS_DIR)
        )
        results = evaluate_sliding_window(model, dataset, config, config.TTA_ENABLED)
    else:
        # Standard evaluation
        transform = get_transform()
        dataset = TestDataset(
            test_df,
            os.path.join(config.DATA_PATH, config.IMAGES_DIR),
            os.path.join(config.DATA_PATH, config.MASKS_DIR),
            config.PATCH_SIZE,
            transform
        )
        loader = DataLoader(
            dataset, batch_size=config.BATCH_SIZE,
            shuffle=False, num_workers=config.NUM_WORKERS
        )
        results = evaluate_standard(model, loader, config)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Dice Score: {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
    print(f"IoU Score:  {results['iou_mean']:.4f} ± {results['iou_std']:.4f}")
    print(f"Recall:     {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
    if 'precision_mean' in results:
        print(f"Precision:  {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
    
    # Visualizations
    full_res_dataset = FullResDataset(
        test_df,
        os.path.join(config.DATA_PATH, config.IMAGES_DIR),
        os.path.join(config.DATA_PATH, config.MASKS_DIR)
    )
    
    visualize_results(
        model, full_res_dataset, config,
        num_samples=args.num_vis,
        output_path=os.path.join(config.OUTPUT_DIR, 'predictions_visualization.png')
    )
    
    plot_metrics_distribution(
        results['all_metrics'],
        os.path.join(config.OUTPUT_DIR, 'metrics_distribution.png')
    )
    
    # Save results
    summary = {
        'dice_mean': float(results['dice_mean']),
        'dice_std': float(results['dice_std']),
        'iou_mean': float(results['iou_mean']),
        'iou_std': float(results['iou_std']),
        'recall_mean': float(results['recall_mean']),
        'recall_std': float(results['recall_std']),
        'model_path': args.model,
        'sliding_window': args.sliding_window,
        'tta': config.TTA_ENABLED
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {config.OUTPUT_DIR}/")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()

