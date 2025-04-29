import cv2
import numpy as np
import time
import logging
from datetime import datetime
import os
import yaml
from ultralytics import YOLO

class WebcamDetector:
    def __init__(self, config_path='config.yaml'):
        """Initialize the webcam detector with configuration"""
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # Initialize the hand detector model with custom weights
        try:
            # Use the last trained weights from your output directory
            weights_path = os.path.join(
                self.config['paths']['output_dir'],
                'runs/pose/train11/weights/best.pt'
            )
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Weights file not found at {weights_path}")
            
            self.model = YOLO(weights_path)
            self.logger.info("Custom hand detection model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def _load_config(self, config_path):
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise

    def initialize_camera(self, camera_id=0):
        """Initialize the webcam"""
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        self.logger.info("Camera initialized successfully")

    def calculate_fps(self):
        """Calculate FPS"""
        self.frame_count += 1
        if (time.time() - self.last_time) > 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = time.time()
        return self.fps

    def draw_detections(self, frame, result):
        """Draw hand detections and wrist keypoints"""
        annotated_frame = frame.copy()
        
        if result.keypoints is not None:
            keypoints = result.keypoints.data
            boxes = result.boxes.data
            
            for box, kpts in zip(boxes, keypoints):
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box[:4])
                conf = float(box[4])
                
                # Draw bounding box for hand
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add confidence score
                cv2.putText(annotated_frame, f"Hand: {conf:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                # Draw wrist keypoint
                for kpt in kpts:
                    x, y = map(int, kpt[:2])
                    conf = float(kpt[2])
                    if conf > 0.5:  # Only draw high-confidence keypoints
                        cv2.circle(annotated_frame, (x, y), 5, (255, 0, 0), -1)
                        cv2.putText(annotated_frame, f"Wrist: {conf:.2f}", 
                                  (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.5, (255, 0, 0), 2)
        
        # Add FPS counter
        fps = self.calculate_fps()
        cv2.putText(annotated_frame, f'FPS: {fps}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame

    def process_frame(self, frame):
        """Process each frame for hand detection"""
        try:
            # Run inference
            results = self.model.predict(
                source=frame,
                conf=0.25,  # Confidence threshold
                iou=0.45,   # NMS IoU threshold
                verbose=False,
                stream=True
            )
            
            # Get the first result
            result = next(results)
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, result)
            
            return annotated_frame
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame

    def run(self):
        """Main loop for webcam detection"""
        try:
            self.initialize_camera()
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.error("Failed to grab frame")
                    break

                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display the frame
                cv2.imshow('Hand-Wrist Detection', processed_frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            self.logger.error(f"Error in webcam detection: {e}")
            raise
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Webcam detection stopped")

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run detector
    detector = WebcamDetector()
    detector.run()

if __name__ == '__main__':
    main()
