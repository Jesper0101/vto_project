# Hand-Wrist Detection for Virtual Try-On (VTO)

A computer vision system that detects hands and wrist keypoints for virtual try-on applications such as watches, bracelets, and other wearables.

![Hand-Wrist Detection Demo](https://github.com/Jesper0101/vto_project/raw/main/demo/demo.gif)

## Overview

This project demonstrates a Virtual Try-On (VTO) system that uses YOLOv8 pose detection models to accurately identify hands and wrist keypoints. It enables precise virtual placement of products on users' wrists through a webcam, creating an interactive virtual try-on experience for accessories.

This repository contains the deployment code and pre-trained model. The training code is proprietary and not included as this is part of a commercial project.

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
git clone https://github.com/Jesper0101/vto_project.git
cd vto_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# Or install dependencies manually:
pip install torch opencv-python ultralytics numpy pyyaml
```

3. The pre-trained model (`best.pt`) is included in the repository.

## Configuration

The project uses two YAML files for configuration:

> **Note:** When using this repository, make sure to update the paths in `config.yaml` to match your local directory structure. The example below uses relative paths which are recommended for portability.


### `config.yaml`

This file contains the main configuration parameters for the model, training and detection system:

```yaml
paths:
  hand_img_dir: "./datasets/hand_wrist/train/images"
  non_hand_dir: "./datasets/hand_wrist/non-hands"        
  annotations_dir: "./datasets/hand_wrist/train/labels"
  output_dir: "./output"
model:
  size: "n"  # Options: n (nano), s (small), m (medium), l (large), x (xlarge)
  epochs: 50  # Number of training epochs
  image_size: 224  # Input image size
  batch_size: 16  # Batch size for training
  pretrained: true  # Use pretrained weights
  conf_thres: 0.25  # Confidence threshold for detection
  iou_thres: 0.45  # IoU threshold for NMS
  device: ""  # Device selection (empty string for auto-selection)
training:
  train_ratio: 0.7
  val_ratio: 0.15
  seed: 42
```

### `data.yaml`

This file defines the dataset structure for training and validation:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images
kpt_shape: [1, 3]
flip_idx: [0]
nc: 1
names: ['hand']
roboflow:
  workspace: hand-isb9j
  project: hand-bmhpi
  version: 2
  license: CC BY 4.0
  url: https://universe.roboflow.com/hand-isb9j/hand-bmhpi/dataset/2
```

## Usage

### Running webcam detection

```bash
python webcam_detector.py
```

This will open your webcam and start detecting hands and wrist keypoints in real-time. Press 'q' to quit.

> **Note:** The training and evaluation scripts are not included in this repository as they contain proprietary code.

## Project Structure

```
vto-project/
├── README.md                # Project documentation
├── config.yaml              # Configuration file
├── data.yaml                # Dataset configuration
├── best.pt                  # Pre-trained model weights
├── hand_wrist_detector.py   # Main detection class
└── webcam_detector.py       # Webcam implementation
```

> **Note:** This repository contains only the deployment code and model files. The training code and dataset processing scripts are proprietary and not included as they are part of a commercial project.
```

## Model Training Details

The hand-wrist detection model is trained using YOLOv8's pose estimation capabilities with the following approach:

1. **Base Model**: YOLOv8-nano pose model pre-trained on COCO keypoints
2. **Custom Dataset**: Created from Roboflow's hand detection dataset with custom wrist keypoint annotations
3. **Fine-tuning**: Model trained for 50 epochs with optimized hyperparameters
4. **Data Augmentation**: Rotation, scaling, and flipping to improve model robustness
5. **Keypoint Configuration**: Single keypoint (wrist) with [1,3] shape and appropriate flip indexes

The model is specifically designed to accurately locate the wrist position, which is crucial for realistic placement of virtual accessories in the VTO application.

## Performance

The model achieves:
- **FPS**: 20-30 FPS on CPU, 60+ FPS on GPU
- **Hand detection accuracy**: 0.92 mAP50
- **Wrist keypoint precision**: 0.87 OKS

The model is optimized for real-time detection with a small footprint, making it suitable for deployment in web applications and on devices with limited computational resources.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- This project uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection and pose estimation
- Thanks to [OpenCV](https://opencv.org/) for computer vision utilities
