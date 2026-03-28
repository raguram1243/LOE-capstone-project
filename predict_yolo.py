"""
YOLO Inference Script
Test/predict using YOLO11 or YOLOv10 trained models
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import torch

def predict_images(
    model_path,
    source_path,
    conf_threshold=0.25,
    iou_threshold=0.45,
    save_results=True,
    save_txt=False,
    save_conf=False,
    output_dir='predictions'
):
    """
    Run inference on images using trained YOLO model
    
    Args:
        model_path: Path to trained model weights (.pt file)
        source_path: Path to image/video/folder
        conf_threshold: Confidence threshold (0-1)
        iou_threshold: IOU threshold for NMS (not used in YOLOv10)
        save_results: Save prediction images
        save_txt: Save results as .txt files
        save_conf: Include confidence in saved results
        output_dir: Directory to save results
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Check if source exists
    if not os.path.exists(source_path):
        print(f"Error: Source not found at {source_path}")
        return
    
    # Determine model type (check if YOLOv10 - NMS-free)
    model_name = Path(model_path).parent.parent.name
    is_yolov10 = 'yolov10' in model_name.lower() or 'v10' in model_name.lower()
    
    print(f"\nModel Type: {'YOLOv10 (NMS-free)' if is_yolov10 else 'YOLO11'}")
    print(f"Source: {source_path}")
    print(f"Confidence Threshold: {conf_threshold}")
    if not is_yolov10:
        print(f"IOU Threshold: {iou_threshold}")
    print(f"Save Results: {save_results}")
    print(f"Output Directory: {output_dir}\n")
    
    # Prepare prediction arguments
    predict_args = {
        'source': source_path,
        'conf': conf_threshold,
        'save': save_results,
        'save_txt': save_txt,
        'save_conf': save_conf,
        'project': output_dir,
        'name': 'predict',
        'exist_ok': True,
        'show_labels': True,
        'show_conf': True,
        'line_width': 2,
    }
    
    # Add IOU threshold only for non-YOLOv10 models
    if not is_yolov10:
        predict_args['iou'] = iou_threshold
    
    # Run prediction
    print("Running inference...")
    results = model.predict(**predict_args)
    
    # Process and display results
    print(f"\nProcessed {len(results)} image(s)")
    
    for i, result in enumerate(results):
        print(f"\n--- Image {i+1} ---")
        print(f"Path: {result.path}")
        print(f"Shape: {result.orig_shape}")
        
        # Get detection info
        boxes = result.boxes
        if len(boxes) > 0:
            print(f"Detections: {len(boxes)}")
            
            # Count detections per class
            class_counts = {}
            for box in boxes:
                cls = int(box.cls[0])
                cls_name = model.names[cls]
                conf = float(box.conf[0])
                
                if cls_name not in class_counts:
                    class_counts[cls_name] = 0
                class_counts[cls_name] += 1
                
                # Print each detection
                print(f"  - {cls_name}: {conf:.2f}")
            
            print("\nSummary:")
            for cls_name, count in class_counts.items():
                print(f"  {cls_name}: {count}")
        else:
            print("No detections")
    
    print(f"\n{'='*50}")
    print("Inference completed!")
    if save_results:
        print(f"Results saved to: {output_dir}/predict")
    print(f"{'='*50}\n")
    
    return results


def predict_video(
    model_path,
    video_path,
    conf_threshold=0.25,
    iou_threshold=0.45,
    save_video=True,
    output_dir='predictions',
    show_live=False
):
    """
    Run inference on video
    
    Args:
        model_path: Path to trained model weights
        video_path: Path to video file or camera (0 for webcam)
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold (not used in YOLOv10)
        save_video: Save output video
        output_dir: Directory to save results
        show_live: Display live predictions
    """
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Determine model type
    model_name = Path(model_path).parent.parent.name
    is_yolov10 = 'yolov10' in model_name.lower() or 'v10' in model_name.lower()
    
    print(f"Model Type: {'YOLOv10 (NMS-free)' if is_yolov10 else 'YOLO11'}")
    print(f"Processing video: {video_path}\n")
    
    # Prepare prediction arguments
    predict_args = {
        'source': video_path,
        'conf': conf_threshold,
        'save': save_video,
        'project': output_dir,
        'name': 'video_predict',
        'exist_ok': True,
        'stream': True,  # Stream mode for videos
        'show': show_live,
    }
    
    if not is_yolov10:
        predict_args['iou'] = iou_threshold
    
    # Run prediction
    results = model.predict(**predict_args)
    
    # Process results
    for result in results:
        # Results are streamed, process each frame
        pass
    
    print(f"\nVideo processing completed!")
    if save_video:
        print(f"Video saved to: {output_dir}/video_predict")


def predict_realtime(
    model_path,
    camera_id=0,
    conf_threshold=0.25,
    iou_threshold=0.45
):
    """
    Run real-time inference using webcam
    
    Args:
        model_path: Path to trained model weights
        camera_id: Camera device ID (0 for default webcam)
        conf_threshold: Confidence threshold
        iou_threshold: IOU threshold (not used in YOLOv10)
    """
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Determine model type
    model_name = Path(model_path).parent.parent.name
    is_yolov10 = 'yolov10' in model_name.lower()
    
    print(f"Model Type: {'YOLOv10 (NMS-free)' if is_yolov10 else 'YOLO11'}")
    print("Starting real-time detection...")
    print("Press 'q' to quit\n")
    
    # Prepare prediction arguments
    predict_args = {
        'source': camera_id,
        'conf': conf_threshold,
        'show': True,
        'stream': True,
    }
    
    if not is_yolov10:
        predict_args['iou'] = iou_threshold
    
    # Run prediction
    try:
        results = model.predict(**predict_args)
        for result in results:
            pass  # Results are displayed in real-time
    except KeyboardInterrupt:
        print("\nStopped by user")


def compare_models(
    model1_path,
    model2_path,
    test_images_path,
    conf_threshold=0.25
):
    """
    Compare two models side by side
    
    Args:
        model1_path: Path to first model
        model2_path: Path to second model
        test_images_path: Path to test images
        conf_threshold: Confidence threshold
    """
    
    print("="*50)
    print("Model Comparison")
    print("="*50)
    
    # Load models
    print(f"\nModel 1: {model1_path}")
    model1 = YOLO(model1_path)
    
    print(f"Model 2: {model2_path}")
    model2 = YOLO(model2_path)
    
    # Run inference on both models
    print(f"\nRunning inference on: {test_images_path}\n")
    
    print("Model 1 Results:")
    print("-" * 30)
    results1 = predict_images(
        model1_path,
        test_images_path,
        conf_threshold=conf_threshold,
        output_dir='comparison/model1'
    )
    
    print("\nModel 2 Results:")
    print("-" * 30)
    results2 = predict_images(
        model2_path,
        test_images_path,
        conf_threshold=conf_threshold,
        output_dir='comparison/model2'
    )
    
    print("\n" + "="*50)
    print("Comparison completed!")
    print("Check 'comparison' folder for results")
    print("="*50)


if __name__ == "__main__":
    # Example usage
    
    # ===== OPTION 1: Predict on images =====
    print("\n1. Image Prediction")
    predict_images(
        model_path='yolo11_runs/train_exp1/weights/best.pt',  # Update path
        source_path='test_images/',  # Can be image file or folder
        conf_threshold=0.25,
        save_results=True
    )
    
    # ===== OPTION 2: Predict on video =====
    # print("\n2. Video Prediction")
    # predict_video(
    #     model_path='yolo11_runs/train_exp1/weights/best.pt',
    #     video_path='test_video.mp4',
    #     conf_threshold=0.25,
    #     save_video=True
    # )
    
    # ===== OPTION 3: Real-time webcam =====
    # print("\n3. Real-time Webcam")
    # predict_realtime(
    #     model_path='yolo11_runs/train_exp1/weights/best.pt',
    #     camera_id=0,
    #     conf_threshold=0.25
    # )
    
    # ===== OPTION 4: Compare YOLO11 vs YOLOv10 =====
    # print("\n4. Model Comparison")
    # compare_models(
    #     model1_path='yolo11_runs/train_exp1/weights/best.pt',
    #     model2_path='yolov10_runs/train_exp1/weights/best.pt',
    #     test_images_path='test_images/',
    #     conf_threshold=0.25
    # )
