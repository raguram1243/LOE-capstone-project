"""
YOLOv8 Training Script
Trains a YOLOv8 model on your custom dataset
YOLOv8 is the baseline model with excellent balance of speed and accuracy
"""

from ultralytics import YOLO
import torch
import os

def train_yolov8(
    model_size='n',  # n, s, m, l, x
    data_yaml='dataset.yaml',
    epochs=45,
    img_size=416,
    batch_size=4,
    device="cpu",
    project_name='yolov8_runs',
    experiment_name='train'
):
    """
    Train YOLOv8 model
    
    Args:
        model_size: Model variant (n/s/m/l/x)
        data_yaml: Path to dataset YAML file
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Batch size for training
        device: Device to use (None for auto-detect, 'cpu', or GPU number)
        project_name: Project folder name
        experiment_name: Experiment name
    """
    
    # Check if CUDA is available
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model_name = f'yolov8{model_size}.pt'
    print(f"\nLoading {model_name}...")
    
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("The model will be downloaded automatically on first use.")
        model = YOLO(model_name)
    
    # Check if dataset YAML exists
    if not os.path.exists(data_yaml):
        print(f"\nWarning: {data_yaml} not found!")
        print("Please create a dataset.yaml file with your dataset configuration.")
        print("\nExample dataset.yaml:")
        print("""
path: /path/to/yolo_dataset
train: images/train
val: images/val

nc: 3  # number of classes
names: ['class1', 'class2', 'class3']
        """)
        return
    
    # Training parameters
    training_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': img_size,
        'batch': batch_size,
        'device': device,
        'project': project_name,
        'name': experiment_name,
        'patience': 20,  # Early stopping patience
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'cache': False,  # Cache images for faster training (uses more RAM)
        'workers': 4,  # Number of worker threads
        'amp': False,  # Automatic Mixed Precision
        
        # Augmentation parameters
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,    # HSV-Saturation
        'hsv_v': 0.4,    # HSV-Value
        'degrees': 0.0,  # Rotation (+/- deg)
        'translate': 0.1,  # Translation (+/- fraction)
        'scale': 0.5,    # Scaling (+/- gain)
        'shear': 0.0,    # Shear (+/- deg)
        'perspective': 0.0,  # Perspective (+/- fraction)
        'flipud': 0.0,   # Flip up-down (probability)
        'fliplr': 0.5,   # Flip left-right (probability)
        'mosaic': 1.0,   # Mosaic augmentation (probability)
        'mixup': 0.0,    # MixUp augmentation (probability)
        
        # Optimizer parameters
        'optimizer': 'auto',  # optimizer (SGD, Adam, AdamW, auto)
        'lr0': 0.01,     # Initial learning rate
        'lrf': 0.01,     # Final learning rate (lr0 * lrf)
        'momentum': 0.937,  # SGD momentum/Adam beta1
        'weight_decay': 0.0005,  # Optimizer weight decay
        'warmup_epochs': 3.0,  # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup initial momentum
        'warmup_bias_lr': 0.1,  # Warmup initial bias lr
        
        # Validation parameters
        'val': True,  # Validate during training
        'plots': True,  # Save plots during training
        'verbose': True,  # Verbose output
    }
    
    print("\n" + "="*50)
    print("YOLOv8 Training Configuration")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {img_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print("\nYOLOv8 Features:")
    print("  ✓ Anchor-free architecture")
    print("  ✓ Efficient C2f modules")
    print("  ✓ Optimized for speed and accuracy")
    print("="*50 + "\n")
    
    # Train the model
    try:
        print("Starting training...\n")
        results = model.train(**training_args)
        
        print("\n" + "="*50)
        print("Training Completed!")
        print("="*50)
        print(f"Results saved to: {project_name}/{experiment_name}")
        print(f"Best weights: {project_name}/{experiment_name}/weights/best.pt")
        print(f"Last weights: {project_name}/{experiment_name}/weights/last.pt")
        
        # Run validation
        print("\nRunning validation on best model...")
        metrics = model.val()
        
        print("\nValidation Metrics:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        
        return results, metrics
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def resume_training(weights_path='runs/detect/train/weights/last.pt'):
    """Resume training from a checkpoint"""
    print(f"Resuming training from: {weights_path}")
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}")
        return
    
    model = YOLO(weights_path)
    results = model.train(resume=True)
    return results


if __name__ == "__main__":
    # Example usage
    print("YOLOv8 Training Script")
    print("=" * 50 + "\n")
    
    # Option 1: Train from scratch
    results, metrics = train_yolov8(
        model_size='n',  # Options: 'n', 's', 'm', 'l', 'x'
        data_yaml='dataset.yaml',  # Update this path
        epochs=45,
        img_size=416,
        batch_size=4,
        device="cpu",  # Auto-detect (use 'cpu' to force CPU)
        project_name='yolov8_runs',
        experiment_name='train_exp1'
    )
    
    # Option 2: Resume training (uncomment to use)
    # results = resume_training('yolov8_runs/train_exp1/weights/last.pt')
