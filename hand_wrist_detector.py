import os
import onnx
import time
import yaml
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class HandWristDetector:
    def __init__(self, config_path='config.yaml'):
        """
        Initialize HandWristDetector with configuration
        
        Args:
            config_path (str): Path to the configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize YOLO pose detection model
        model_size = self.config['model']['size']
        model_path = f"yolov8{model_size}-pose.pt"
        
        # Download model if not exists
        if not os.path.exists(model_path):
            print(f"Downloading YOLOv8{model_size} pose model...")
        
        self.model = YOLO(model_path)
        
    def train(self, data_yaml):
        """
        Train the model with custom configuration
        
        Args:
            data_yaml (str): Path to the data YAML file containing dataset configuration
            
        Returns:
            results: Training results object
        """
        # Set training arguments
        args = dict(
            data=data_yaml,                    # Path to data YAML file
            task='pose',                       # Task type for pose detection
            mode='train',                      # Training mode
            model=self.model,                  # Model to train
            epochs=self.config['model']['epochs'],
            imgsz=self.config['model']['image_size'],
            batch=self.config['model']['batch_size'],
            device='',                         # Device to use (auto-select)
            workers=8,                         # Number of worker threads
            optimizer='AdamW',                  # Optimizer to use (Adam)
            patience=20,                       # Early stopping patience
            verbose=True,                      # Print verbose output
            seed=0,                           # Random seed
            deterministic=True,                # Enable deterministic mode
            single_cls=True,                   # Single class training
            rect=True,                         # Rectangular training
            cos_lr=True,                       # Cosine learning rate scheduler
            close_mosaic=10,                   # Disable mosaic augmentation for final epochs
            resume=False,                      # Resume training
            amp=True,                          # Automatic Mixed Precision
            
            # Learning rate settings
            lr0=0.001,                        # Initial learning rate
            lrf=0.01,                         # Final learning rate fraction
            momentum=0.937,                    # SGD momentum/Adam beta1
            weight_decay=0.0005,              # Optimizer weight decay
            warmup_epochs=3.0,                # Warmup epochs
            warmup_momentum=0.8,              # Warmup initial momentum
            warmup_bias_lr=0.1,               # Warmup initial bias learning rate
            
            # Loss coefficients
            box=7.5,                          # Box loss gain
            cls=0.5,                          # Class loss gain
            pose=12.0,                        # Pose loss gain
            kobj=2.0,                         # Keypoint obj loss gain
            
            # Augmentation settings
            degrees=10.0,                      # Rotation degrees
            translate=0.2,                    # Translation
            scale=0.7,                        # Scale
            fliplr=0.5,                       # Horizontal flip probability
            mosaic=1.0,                       # Mosaic probability
            mixup=0.0,                        # Mixup probability
            
            # Saving settings
            project='runs/pose',              # Project name
            name='train',                     # Run name
            exist_ok=False,                   # Allow existing project
            pretrained=True,                  # Use pretrained model
            plots=True,                       # Generate plots
            save=True,                        # Save train checkpoints
            save_period=-1,                   # Save checkpoint every x epochs
            
            # Validation settings
            val=True,                         # Validate during training
            save_json=False,                  # Save JSON validation results
            conf=None,                        # Confidence threshold
            iou=0.7,                          # NMS IoU threshold
            max_det=300,                      # Maximum detections per image
            
            # Advanced settings
            fraction=1.0,                     # Dataset fraction to train on
            profile=False,                    # Profile ONNX/TF.js/TensorRT
            overlap_mask=True,                # Masks should overlap during inference
            mask_ratio=4,                     # Mask downsample ratio
            dropout=0.2,                      # Use dropout regularization
            label_smoothing=0.1,              # Label smoothing epsilon
            nbs=64,                          # Nominal batch size
        )
        
        # Start training
        try:
            results = self.model.train(**args)
            return results
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
    
    def evaluate(self, data_yaml):
        """
        Evaluate the model on validation/test set
        
        Args:
            data_yaml (str): Path to the data YAML file
            
        Returns:
            results: Validation results object
        """
        try:
            results = self.model.val(
                data=data_yaml,
                imgsz=self.config['model']['image_size'],
                batch=self.config['model']['batch_size'],
                conf=0.25,
                iou=0.7,
                device='',
                verbose=True,
                save_json=False,
                save_hybrid=False,
                max_det=300,
                half=False
            )
            return results
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            raise
    
    def export_model(self, format='onnx'):
        """
        Export the model to specified format
        
        Args:
            format (str): Format to export to ('onnx' or 'tflite')
        """
        try:
            if format == 'onnx':
                self.model.export(
                    format='onnx',
                    dynamic=True,
                    simplify=True,
                    opset=11,
                    device='cpu'
                )
            elif format == 'tflite':
                self.model.export(
                    format='tflite',
                    int8=True,
                    device='cpu'
                )
        except Exception as e:
            print(f"Export error: {str(e)}")
            raise
    
    def predict(self, image_path):
        """
        Run inference on a single image
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            results: Detection results object
        """
        try:
            results = self.model.predict(
                source=image_path,
                conf=0.25,
                iou=0.45,
                imgsz=self.config['model']['image_size'],
                device='',
                verbose=False,
                save=True,
                save_txt=False,
                save_conf=False,
                save_crop=False,
                show_labels=True,
                show_conf=True,
                max_det=300,
                agnostic_nms=False,
                classes=None,
                retina_masks=False,
                boxes=True
            )
            return results[0]
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise
    
    def predict_batch(self, image_paths):
        """
        Run inference on a batch of images
        
        Args:
            image_paths (list): List of paths to input images
            
        Returns:
            results: List of detection results objects
        """
        try:
            results = self.model.predict(
                source=image_paths,
                conf=0.25,
                iou=0.45,
                imgsz=self.config['model']['image_size'],
                batch=self.config['model']['batch_size']
            )
            return results
        except Exception as e:
            print(f"Batch prediction error: {str(e)}")
            raise
