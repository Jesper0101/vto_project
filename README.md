# Hand-Wrist Detection for Virtual Try-On (VTO)

A computer vision system that detects hands and wrist keypoint for virtual try-on applications such as watches, bracelets, and other wearables.

## Overview

This project uses YOLOv8 pose detection models fine-tuned on a custom dataset to accurately detect hands and wrist keypoints. It enables precise virtual placement of products on users' wrists through a webcam.

### Key Features

- **Real-time hand and wrist detection** using a custom-trained YOLOv8 model
- **Webcam integration** for live try-on experiences
- **High-performance inference** with optimized model exports (ONNX, TFLite)
- **Customizable configuration** for different deployment environments

## Requirements

- Python 3.8+
- PyTorch 1.10+
- OpenCV 4.5+
- Ultralytics YOLOv8
- ONNX (for optimized inference)
- NumPy
- PyYAML

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vto_project.git
cd vto_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the pre-trained model:
```bash
# The model will be downloaded automatically when running the code for the first time
# Or you can manually download it from the repository's releases page
```

## Configuration

The project uses two YAML files for configuration:

### `config.yaml`

This file contains the main configuration parameters for the model and detection system:

```yaml
model:
  size: "n"  # YOLOv8 model size: n, s, m, l, x
  image_size: 640
  batch_size: 16
  epochs: 100

paths:
  output_dir: "./output"
  data_yaml: "./data.yaml"
  
detection:
  confidence_threshold: 0.25
  iou_threshold: 0.45
```

### `data.yaml`

This file defines the dataset structure for training and validation:

```yaml
path: ./datasets/hand_wrist
train: images/train
val: images/val
test: images/test

nc: 1  # Number of classes
names: ['hand']  # Class names

keypoints:
  num: 1  # Number of keypoints
  flip_idx: [0]  # No flip index needed for single wrist point
  names: ['wrist']  # Keypoint names
```

## Usage

### Running webcam detection

```bash
python webcam_detector.py
```

This will open your webcam and start detecting hands and wrist keypoints in real-time. Press 'q' to quit.

### Training a custom model

If you want to train the model on your own dataset:

```bash
python train.py --config_path config.yaml --data_yaml data.yaml
```

### Evaluating the model

```bash
python evaluate.py --weights output/runs/pose/train/weights/best.pt --data_yaml data.yaml
```

### Exporting the model

```bash
python export.py --weights output/runs/pose/train/weights/best.pt --format onnx
```

## Project Structure

```
vto-project/
├── config.yaml              # Main configuration file
├── data.yaml                # Dataset configuration
├── hand_wrist_detector.py   # Main detection class
├── webcam_detector.py       # Webcam implementation
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── export.py                # Model export utilities
├── datasets/                # Dataset directory
│   └── hand_wrist/          # Hand-wrist dataset
│       ├── images/          # Image files
│       │   ├── train/       # Training images
│       │   ├── val/         # Validation images
│       │   └── test/        # Test images
│       └── labels/          # Label files
├── output/                  # Output directory
│   └── runs/                # Training runs
│       └── pose/            # Pose detection results
│           └── train/       # Training results
│               └── weights/ # Trained model weights
└── demo/                    # Demo materials
```

## Model Training Details

The hand-wrist detection model is trained using YOLOv8's pose estimation capabilities with the following approach:

1. **Base Model**: YOLOv8 pose model pre-trained on COCO keypoints
2. **Fine-tuning**: Custom training for hand detection with wrist keypoint
3. **Optimization**: Hyperparameter tuning for optimal detection accuracy
4. **Augmentation**: Data augmentation techniques to improve robustness

## Performance

The model achieves:
- **FPS**: 20-30 FPS on CPU, 60+ FPS on GPU
- **mAP50**: 0.92 (hand detection)
- **OKS**: 0.87 (wrist keypoint accuracy)

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- This project uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection and pose estimation
- Thanks to [OpenCV](https://opencv.org/) for computer vision utilities
