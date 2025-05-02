"""
Combined script for complete sclera segmentation model evaluation.
This script merges functionality from:
- evaluate_predictions.py
- generate_predictions.py
- check_submission.py
- and all supporting utility modules
"""

import os
import cv2
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import precision_recall_curve, auc, f1_score
import shutil
import warnings
warnings.filterwarnings("ignore")

#############################################
# CONFIGURATION SETTINGS
#############################################

class EvalConfig:
    # Evaluation settings
    PREDICTION_DIR = "predictions"
    GROUND_TRUTH_DIR = "data/test_datasets"  # Default path
    RESULT_DIR = "evaluation_results"
    DATASETS = ["MOBIUS", "SMD+SLD", "Synthetic"]
    VISUALIZE_RESULTS = True
    SAVE_PR_CURVES = True
    
class PredictConfig:
    # Prediction generation settings
    MODEL_PATH = "checkpoints/best_model.pth"
    TEST_DATA_DIR = "data/test_datasets"  # Default path
    OUTPUT_DIR = "predictions"
    NUM_CLASSES = 1  # Binary segmentation (1 class) - CHANGED FROM 4
    BINARY_THRESHOLD = 0.5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    IMAGE_SIZE = (512, 512)
    SAVE_VISUALIZATION = True

#############################################
# MODEL ARCHITECTURE
#############################################

class UNet(torch.nn.Module):
    """UNet architecture matching the checkpoint"""
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder blocks - keep these the same
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True)
        )
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(inplace=True)
        )
        
        # Decoder blocks - CORRECTED based on weight shapes in checkpoint
        # The key insight is that original model used Conv2d instead of ConvTranspose2d with different input dimensions
        
        # Dec4: Expected input [512, 1536, 3, 3] - indicates concatenation with more channels
        self.dec4 = torch.nn.Sequential(
            torch.nn.Conv2d(1024 + 512, 512, kernel_size=3, padding=1),  # 1536 input channels
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True)
        )
        
        # Dec3: Expected input [256, 768, 3, 3]
        self.dec3 = torch.nn.Sequential(
            torch.nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1),  # 768 input channels
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )
        
        # Dec2: Expected input [128, 384, 3, 3]
        self.dec2 = torch.nn.Sequential(
            torch.nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),  # 384 input channels
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )
        
        # Dec1: Expected input [64, 192, 3, 3]
        self.dec1 = torch.nn.Sequential(
            torch.nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),  # 192 input channels
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        # Upsamplers - separate components rather than part of dec blocks
        self.up4 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up1 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final layer - binary output
        self.final = torch.nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections - use Upsample instead of ConvTranspose
        dec4_up = self.up4(bottleneck)
        dec4 = torch.cat([dec4_up, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3_up = self.up3(dec4)
        dec3 = torch.cat([dec3_up, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2_up = self.up2(dec3)
        dec2 = torch.cat([dec2_up, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1_up = self.up1(dec2)
        dec1 = torch.cat([dec1_up, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Final layer
        final = self.final(dec1)
        
        return {"out": final}

def initialize_model(num_classes, architecture="auto", keep_feature_extract=False, use_pretrained=True):
    """Initialize segmentation model"""
    if architecture == "unet" or (architecture == "auto" and num_classes <= 4):
        # Use UNet for sclera segmentation (matches the checkpoint architecture)
        print(f"Initializing UNet model with {num_classes} output classes")
        model = UNet(in_channels=3, out_channels=num_classes)
        
    else:
        # Use DeepLabV3 (the original architecture in the code)
        print(f"Initializing DeepLabV3 model with {num_classes} output classes")
        if use_pretrained: 
            model_deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet101(
                weights="DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1", progress=True)
        else: 
            model_deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet101(
                weights=None, progress=True)
            
        # Modify classifier for sclera segmentation
        model_deeplabv3.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
        
        # Set feature extraction mode if needed
        if keep_feature_extract:
            for param in model_deeplabv3.backbone.parameters():
                param.requires_grad = False
                
        model = model_deeplabv3
        
    return model

#############################################
# DATA LOADING
#############################################

class DataLoaderSegmentation(torch.utils.data.dataset.Dataset):
    """Dataset class for segmentation data loading"""
    def __init__(self, folder_path, mode="test", num_classes=4, image_size=(512, 512)):
        self.img_files = glob.glob(os.path.join(folder_path, 'Images', '*.*'))
        self.label_files = []
        self.mode = mode
        self.num_classes = num_classes
        self.image_size = image_size
        
        for img_path in self.img_files:
            img_name = os.path.basename(img_path)
            if os.path.exists(os.path.join(folder_path, 'Masks', img_name)):
                self.label_files.append(os.path.join(folder_path, 'Masks', img_name))
            else:
                # For test data without masks
                self.label_files.append(None)
                
        # Set transforms
        self.transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.target_transforms = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])
        
    def __getitem__(self, index):
        # Load and transform image
        img_path = self.img_files[index]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transforms(img)
        
        # Get image name for reference
        img_name = os.path.basename(img_path)
        
        # Handle labels if available
        label_path = self.label_files[index]
        if label_path and os.path.exists(label_path):
            label = Image.open(label_path).convert('L')
            label_tensor = self.target_transforms(label)
            
            if self.num_classes == 2:  # Binary segmentation
                # Convert any non-zero value to 1 (sclera)
                label_tensor = (label_tensor > 0).float()
            else:  # Multi-class segmentation
                # Keep original classes
                label_tensor = label_tensor.long().squeeze(0)
        else:
            # For test data without ground truth
            label_tensor = torch.zeros((self.image_size[0], self.image_size[1]))
            
        return img_tensor, label_tensor, img_name
    
    def __len__(self):
        return len(self.img_files)

#############################################
# PREDICTION GENERATION
#############################################

def generate_predictions(model, dataset_dir, output_dir, config):
    """Generate binary and probability map predictions from model"""
    # Make sure output directories exist
    binary_dir = os.path.join(output_dir, "binary_masks")
    prob_dir = os.path.join(output_dir, "probability_maps")
    os.makedirs(binary_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)
    
    # Setup model
    model.eval()
    
    # Create dataset and dataloader
    dataset = DataLoaderSegmentation(
        folder_path=dataset_dir,
        mode="test",
        num_classes=config.NUM_CLASSES,
        image_size=config.IMAGE_SIZE
    )
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    with torch.no_grad():
        for images, _, img_names in tqdm(dataloader, desc="Generating predictions"):
            # Move to device
            images = images.to(config.DEVICE)
            
            # Forward pass
            outputs = model(images)["out"]
            
            # Process each image in batch
            for i, output in enumerate(outputs):
                if config.NUM_CLASSES == 1:
                    # Single channel binary output (sigmoid instead of softmax)
                    prob_map = torch.sigmoid(output[0])  # For single-channel binary output
                    binary_mask = (prob_map > config.BINARY_THRESHOLD).float()
                elif config.NUM_CLASSES == 2:
                    # Binary segmentation with two channels
                    prob_map = torch.softmax(output, dim=0)[1]  # Probability of sclera class
                    binary_mask = (prob_map > config.BINARY_THRESHOLD).float()
                else:
                    # Multi-class segmentation - convert to binary sclera mask
                    # Class 1 is sclera in multi-class setup
                    probs = torch.softmax(output, dim=0)
                    prob_map = probs[1]  # Probability of sclera class
                    binary_mask = (torch.argmax(output, dim=0) == 1).float()
                
                # Convert to numpy for saving
                binary_np = binary_mask.cpu().numpy() * 255
                prob_np = prob_map.cpu().numpy() * 255
                
                # Save outputs
                img_name = img_names[i]
                cv2.imwrite(os.path.join(binary_dir, img_name), binary_np.astype(np.uint8))
                cv2.imwrite(os.path.join(prob_dir, img_name), prob_np.astype(np.uint8))
                
    print(f"Predictions saved to {output_dir}")

#############################################
# METRIC COMPUTATION
#############################################

def calculate_iou(y_true, y_pred):
    """Calculate Intersection over Union (IoU)"""
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 0.0
    return intersection / union

def calculate_metrics(ground_truth_dir, prediction_dir, dataset_name):
    """Calculate F1, IoU and PR-curve metrics for a dataset"""
    gt_path = os.path.join(ground_truth_dir, dataset_name, "Masks")
    binary_pred_path = os.path.join(prediction_dir, dataset_name, "binary_masks")
    prob_pred_path = os.path.join(prediction_dir, dataset_name, "probability_maps")
    
    # Get all ground truth files
    gt_files = glob.glob(os.path.join(gt_path, "*.*"))
    
    # Initialize metric lists
    ious, f1_scores = [], []
    all_gt_flat, all_prob_flat = [], []
    
    for gt_file in gt_files:
        file_name = os.path.basename(gt_file)
        pred_file = os.path.join(binary_pred_path, file_name)
        prob_file = os.path.join(prob_pred_path, file_name)
        
        # Skip if prediction doesn't exist
        if not os.path.exists(pred_file) or not os.path.exists(prob_file):
            print(f"Warning: Prediction missing for {file_name}")
            continue
        
        # Load files
        gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
        prob_mask = cv2.imread(prob_file, cv2.IMREAD_GRAYSCALE)
        
        # Resize if dimensions don't match
        if gt_mask.shape != pred_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
            prob_mask = cv2.resize(prob_mask, (gt_mask.shape[1], gt_mask.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
        
        # Binarize masks
        gt_binary = gt_mask > 0
        pred_binary = pred_mask > 0
        
        # Calculate metrics for this image
        iou = calculate_iou(gt_binary, pred_binary)
        f1 = f1_score(gt_binary.flatten(), pred_binary.flatten(), zero_division=0)
        
        ious.append(iou)
        f1_scores.append(f1)
        
        # Collect data for PR curve
        all_gt_flat.extend(gt_binary.flatten())
        all_prob_flat.extend((prob_mask / 255.0).flatten())
    
    # Calculate overall metrics
    avg_iou = np.mean(ious) if ious else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0
    
    # Calculate precision-recall curve data
    precision, recall, thresholds = precision_recall_curve(all_gt_flat, all_prob_flat)
    pr_auc = auc(recall, precision)
    
    # Find best F1 score and threshold
    f1_scores_pr = [(2 * p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
    best_f1_idx = np.argmax(f1_scores_pr)
    best_f1 = f1_scores_pr[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    
    # Return all metrics
    metrics = {
        "Mean IoU": avg_iou,
        "Mean F1": avg_f1,
        "PR AUC": pr_auc,
        "Best F1": best_f1,
        "Best Threshold": best_threshold,
        "PR Data": (precision, recall)
    }
    
    return metrics

def visualize_pr_curve(precision, recall, auc_score, best_f1, dataset_name, save_path=None):
    """Plot and save precision-recall curve"""
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {auc_score:.4f})')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Classifier')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name} (Best F1: {best_f1:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

#############################################
# SUBMISSION CHECKING
#############################################

def check_submission_format(submission_dir, test_datasets):
    """Validate that the submission format meets requirements"""
    errors = []
    warnings = []
    
    # Check top-level directories
    for dataset in test_datasets:
        dataset_dir = os.path.join(submission_dir, dataset)
        
        # Check dataset directory exists
        if not os.path.exists(dataset_dir):
            errors.append(f"Missing dataset directory: {dataset}")
            continue
        
        # Check sub-directories
        binary_dir = os.path.join(dataset_dir, "binary_masks")
        prob_dir = os.path.join(dataset_dir, "probability_maps")
        
        if not os.path.exists(binary_dir):
            errors.append(f"Missing binary_masks directory for {dataset}")
        
        if not os.path.exists(prob_dir):
            errors.append(f"Missing probability_maps directory for {dataset}")
        
        # Check file counts match
        if os.path.exists(binary_dir) and os.path.exists(prob_dir):
            binary_files = glob.glob(os.path.join(binary_dir, "*.*"))
            prob_files = glob.glob(os.path.join(prob_dir, "*.*"))
            
            if len(binary_files) != len(prob_files):
                errors.append(f"File count mismatch for {dataset}: binary={len(binary_files)}, prob={len(prob_files)}")
            
            # Check file formats
            for bin_file in binary_files:
                file_name = os.path.basename(bin_file)
                prob_file = os.path.join(prob_dir, file_name)
                
                if not os.path.exists(prob_file):
                    errors.append(f"Missing probability file for {file_name} in {dataset}")
                    continue
                
                # Check binary mask is actually binary
                bin_mask = cv2.imread(bin_file, cv2.IMREAD_GRAYSCALE)
                unique_values = np.unique(bin_mask)
                if not np.all(np.isin(unique_values, [0, 255])):
                    warnings.append(f"Binary mask {file_name} in {dataset} contains non-binary values: {unique_values}")
                
                # Check probability map range
                prob_mask = cv2.imread(prob_file, cv2.IMREAD_GRAYSCALE)
                if np.max(prob_mask) == 0:
                    warnings.append(f"Probability map {file_name} in {dataset} is all zeros")
    
    # Print summary
    if errors:
        print("❌ SUBMISSION FORMAT ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print(f"Total errors: {len(errors)}")
    
    if warnings:
        print("⚠️ SUBMISSION FORMAT WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
        print(f"Total warnings: {len(warnings)}")
    
    if not errors and not warnings:
        print("✅ Submission format validation passed!")
    
    return len(errors) == 0

#############################################
# VISUALIZATION
#############################################

def visualize_results(image_path, gt_path, pred_path, prob_path, save_path=None):
    """Create visualization of prediction results"""
    # Load images
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if gt_path else None
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    prob_map = cv2.imread(prob_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize masks to match image if necessary
    if image.shape[:2] != pred_mask.shape[:2]:
        pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        prob_map = cv2.resize(prob_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        if gt_mask is not None:
            gt_mask = cv2.resize(gt_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create overlay masks
    overlay_color = np.array([0, 255, 0], dtype=np.uint8)  # Green for visualization
    
    pred_overlay = image.copy()
    pred_mask_binary = pred_mask > 0
    pred_overlay[pred_mask_binary] = pred_overlay[pred_mask_binary] * 0.5 + overlay_color * 0.5
    
    # Create heat map for probability
    prob_heat = cv2.applyColorMap(prob_map, cv2.COLORMAP_JET)
    prob_heat = cv2.cvtColor(prob_heat, cv2.COLOR_BGR2RGB)
    prob_overlay = cv2.addWeighted(image, 0.7, prob_heat, 0.3, 0)
    
    # Create GT overlay if GT exists
    if gt_mask is not None:
        gt_overlay = image.copy()
        gt_mask_binary = gt_mask > 0
        gt_overlay[gt_mask_binary] = gt_overlay[gt_mask_binary] * 0.5 + overlay_color * 0.5
        
        # Calculate error map (FP and FN)
        error_map = np.zeros_like(image)
        # False positives (red)
        error_map[(pred_mask_binary) & (~gt_mask_binary)] = [255, 0, 0]
        # False negatives (blue)
        error_map[(~pred_mask_binary) & (gt_mask_binary)] = [0, 0, 255]
        
        # Plot all visualizations
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(gt_overlay)
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(pred_overlay)
        plt.title('Prediction')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(prob_overlay)
        plt.title('Probability Map')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(error_map)
        plt.title('Error Map (Red=FP, Blue=FN)')
        plt.axis('off')
    else:
        # Plot without ground truth
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(pred_overlay)
        plt.title('Prediction')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(prob_overlay)
        plt.title('Probability Map')
        plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

#############################################
# MAIN EVALUATION FLOW
#############################################

def run_full_evaluation(args):
    """Main function to run the complete evaluation pipeline"""
    print("="*80)
    print("SCLERA SEGMENTATION EVALUATION PIPELINE")
    print("="*80)
    
    # Create configurations
    eval_config = EvalConfig()
    predict_config = PredictConfig()
    
    # Override with command-line args if provided
    if args.model_path:
        predict_config.MODEL_PATH = args.model_path
    if args.output_dir:
        predict_config.OUTPUT_DIR = args.output_dir
        eval_config.PREDICTION_DIR = args.output_dir
    if args.test_data_dir:
        predict_config.TEST_DATA_DIR = args.test_data_dir
        eval_config.GROUND_TRUTH_DIR = args.test_data_dir
    
    # 1. Load model
    print("\n[1] Loading model...")
    try:
        # Load the checkpoint
        checkpoint = torch.load(predict_config.MODEL_PATH, map_location=predict_config.DEVICE)
        print(f"Checkpoint loaded from {predict_config.MODEL_PATH}")
        
        # Check if it's a dictionary (checkpoint) or a direct model
        if isinstance(checkpoint, dict):
            print("Checkpoint is a dictionary, extracting model...")
            
            # Initialize model with the correct architecture
            model = initialize_model(num_classes=predict_config.NUM_CLASSES, architecture="unet")
            
            # Common keys in checkpoint dictionaries
            model_keys = ['model', 'state_dict', 'model_state_dict', 'net', 'network']
            
            # Find the right key for the model state dict
            state_dict = None
            for key in model_keys:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    print(f"Found model state dict under key: '{key}'")
                    break
            
            # If no known keys found, try using the whole dict if it looks like a state dict
            if state_dict is None:
                # Check if any keys in the dict match model parameter names
                model_param_names = [name for name, _ in model.named_parameters()]
                if any(key in model_param_names or key.replace('module.', '') in model_param_names 
                       for key in checkpoint.keys()):
                    state_dict = checkpoint
                    print("Using entire checkpoint as state dict")
                else:
                    # Print available keys to help debugging
                    print(f"Available keys in checkpoint: {list(checkpoint.keys())}")
                    raise ValueError("Could not find model state dict in checkpoint")
            
            # Check if state dict has "module." prefix (common with DataParallel)
            if any(key.startswith('module.') for key in state_dict.keys()):
                print("Removing 'module.' prefix from state dict keys")
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load state dict into model
            model.load_state_dict(state_dict, strict=False)
            print("Model state dict loaded successfully")
            
        else:
            # The loaded object is already a model
            model = checkpoint
            print("Loaded object is a direct model instance")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nPossible solutions:")
        print("1. Check if the model architecture matches what was used during training")
        print("2. Verify the checkpoint format (open it with torch.load in a separate script)")
        print("3. Try using a different checkpoint file")
        exit(1)

    model = model.to(predict_config.DEVICE)
    
    # 2. Generate predictions for all datasets
    print("\n[2] Generating predictions...")
    for dataset in eval_config.DATASETS:
        dataset_dir = os.path.join(predict_config.TEST_DATA_DIR, dataset)
        output_dir = os.path.join(predict_config.OUTPUT_DIR, dataset)
        
        if not os.path.exists(dataset_dir):
            print(f"⚠️ Dataset not found: {dataset_dir}")
            continue
            
        print(f"Processing dataset: {dataset}")
        generate_predictions(model, dataset_dir, output_dir, predict_config)
    
    # 3. Evaluate predictions
    print("\n[3] Evaluating predictions...")
    os.makedirs(eval_config.RESULT_DIR, exist_ok=True)
    
    # Store metrics for all datasets
    all_metrics = {}
    
    for dataset in eval_config.DATASETS:
        gt_dir = eval_config.GROUND_TRUTH_DIR
        pred_dir = eval_config.PREDICTION_DIR
        
        # Skip evaluation if ground truth is missing
        if not os.path.exists(os.path.join(gt_dir, dataset, "Masks")):
            print(f"⚠️ Ground truth not found for {dataset}, skipping evaluation")
            continue
            
        print(f"Evaluating dataset: {dataset}")
        metrics = calculate_metrics(gt_dir, pred_dir, dataset)
        all_metrics[dataset] = metrics
        
        # Print metrics
        print(f"  Mean IoU: {metrics['Mean IoU']:.4f}")
        print(f"  Mean F1: {metrics['Mean F1']:.4f}")
        print(f"  PR-AUC: {metrics['PR AUC']:.4f}")
        print(f"  Best F1: {metrics['Best F1']:.4f}")
        print(f"  Best Threshold: {metrics['Best Threshold']:.4f}")
        
        # Save PR curve if enabled
        if eval_config.SAVE_PR_CURVES:
            pr_curve_path = os.path.join(eval_config.RESULT_DIR, f"pr_curve_{dataset}.png")
            visualize_pr_curve(
                metrics['PR Data'][0], metrics['PR Data'][1], 
                metrics['PR AUC'], metrics['Best F1'],
                dataset, pr_curve_path
            )
    
    # 4. Check submission format
    print("\n[4] Checking submission format...")
    check_submission_format(eval_config.PREDICTION_DIR, eval_config.DATASETS)
    
    # 5. Visualize some results if enabled
    if eval_config.VISUALIZE_RESULTS:
        print("\n[5] Generating visualizations...")
        os.makedirs(os.path.join(eval_config.RESULT_DIR, "visualizations"), exist_ok=True)
        
        for dataset in eval_config.DATASETS:
            # Skip if dataset doesn't exist
            if not os.path.exists(os.path.join(eval_config.GROUND_TRUTH_DIR, dataset)):
                continue
                
            # Get images to visualize (max 5 per dataset)
            img_files = glob.glob(os.path.join(eval_config.GROUND_TRUTH_DIR, dataset, "Images", "*.*"))
            if not img_files:
                continue
                
            # Sample a few images to visualize
            sample_size = min(5, len(img_files))
            samples = np.random.choice(img_files, sample_size, replace=False)
            
            for img_path in samples:
                img_name = os.path.basename(img_path)
                gt_path = os.path.join(eval_config.GROUND_TRUTH_DIR, dataset, "Masks", img_name)
                pred_path = os.path.join(eval_config.PREDICTION_DIR, dataset, "binary_masks", img_name)
                prob_path = os.path.join(eval_config.PREDICTION_DIR, dataset, "probability_maps", img_name)
                
                if not os.path.exists(gt_path) or not os.path.exists(pred_path) or not os.path.exists(prob_path):
                    continue
                    
                save_path = os.path.join(eval_config.RESULT_DIR, "visualizations", f"{dataset}_{img_name}")
                visualize_results(img_path, gt_path, pred_path, prob_path, save_path)
    
    # 6. Print final summary
    print("\n[6] Evaluation Summary:")
    print("-" * 60)
    print(f"{'Dataset':<15} {'IoU':<10} {'F1':<10} {'PR-AUC':<10} {'Best F1':<10}")
    print("-" * 60)
    
    overall_iou, overall_f1, overall_auc, overall_best_f1 = [], [], [], []
    
    for dataset, metrics in all_metrics.items():
        print(f"{dataset:<15} {metrics['Mean IoU']:<10.4f} {metrics['Mean F1']:<10.4f} "
              f"{metrics['PR AUC']:<10.4f} {metrics['Best F1']:<10.4f}")
        
        overall_iou.append(metrics['Mean IoU'])
        overall_f1.append(metrics['Mean F1'])
        overall_auc.append(metrics['PR AUC'])
        overall_best_f1.append(metrics['Best F1'])
    
    print("-" * 60)
    print(f"{'OVERALL':<15} {np.mean(overall_iou):<10.4f} {np.mean(overall_f1):<10.4f} "
          f"{np.mean(overall_auc):<10.4f} {np.mean(overall_best_f1):<10.4f}")
    print("-" * 60)
    
    print(f"\nResults saved to {eval_config.RESULT_DIR}")
    print("\nEvaluation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sclera Segmentation Evaluation")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, help="Output directory for predictions")
    parser.add_argument("--test_data_dir", type=str, help="Directory containing test datasets")
    args = parser.parse_args()
    
    run_full_evaluation(args)