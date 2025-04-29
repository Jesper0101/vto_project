import os
import cv2
import yaml
from sklearn.model_selection import train_test_split

class HandDatasetPreprocessor:
    def __init__(self, config_path='config.yaml', hand_img_dir=None, annotations_dir=None, 
                 non_hand_dir=None, max_samples=6000, max_negative_samples=500):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.hand_img_dir = hand_img_dir
        self.annotations_dir = annotations_dir
        self.non_hand_dir = non_hand_dir
        self.output_dir = os.path.abspath(self.config['paths']['output_dir'])
        self.dataset_dir = os.path.join(self.output_dir, 'hand_dataset')
        self.max_samples = max_samples
        self.max_negative_samples = max_negative_samples
        
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.dataset_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_dir, split, 'labels'), exist_ok=True)

    def _get_negative_samples(self):
        """Gets list of negative sample images."""
        if not self.non_hand_dir:
            return []
        
        negative_samples = []
        valid_extensions = ('.jpg', '.jpeg', '.png')
        
        for file in os.listdir(self.non_hand_dir):
            if file.lower().endswith(valid_extensions):
                img_path = os.path.join(self.non_hand_dir, file)
                try:
                    # Verify the image can be opened
                    image = cv2.imread(img_path)
                    if image is not None:
                        negative_samples.append({
                            'image_path': img_path,
                            'is_negative': True
                        })
                except Exception as e:
                    print(f"Error reading negative sample {file}: {e}")
        
        print(f"Found {len(negative_samples)} negative samples.")
        return negative_samples[:self.max_negative_samples]

    def _get_image_annotation_pairs(self):
        """Pairs each image with its annotation file."""
        if not self.hand_img_dir or not self.annotations_dir:
            raise ValueError("Image and annotations directories must be specified.")
        
        pairs = []
        for file in os.listdir(self.hand_img_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(self.hand_img_dir, file)
                annotation_name = os.path.splitext(file)[0] + '.txt'
                annotation_path = os.path.join(self.annotations_dir, annotation_name)
                
                if os.path.exists(annotation_path):
                    pairs.append((img_path, annotation_path))
                else:
                    print(f"Warning: No annotation found for {file}")
        
        print(f"Found {len(pairs)} positive image-annotation pairs.")
        return pairs[:self.max_samples]

    def _validate_keypoints(self, line):
        """Validates the simplified keypoint format (8 values)."""
        try:
            parts = line.strip().split()
            
            # Check if we have exactly 8 values
            if len(parts) != 8:
                print(f"Invalid number of values in line. Expected 8, got {len(parts)}")
                return None
            
            # Parse values
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            keypoint = [float(x) for x in parts[5:8]]
            
            # Validate class ID
            if class_id != 0:
                print(f"Invalid class ID: {class_id}")
                return None
                
            # Validate bbox values are between 0 and 1
            if not all(0 <= x <= 1 for x in bbox):
                print(f"Invalid bbox values (must be between 0 and 1): {bbox}")
                return None
            
            # Validate keypoint x,y values are between 0 and 1
            if not all(0 <= x <= 1 for x in keypoint[:2]):
                print(f"Invalid keypoint coordinates (must be between 0 and 1): {keypoint[:2]}")
                return None
            
            return line.strip()
            
        except (ValueError, IndexError) as e:
            print(f"Error processing keypoint line: {e}")
            return None

    def _prepare_samples(self):
        """Prepares samples with validated keypoint annotations and negative samples."""
        print("Preparing samples...")
        pairs = self._get_image_annotation_pairs()
        negative_samples = self._get_negative_samples()
        samples = []
        invalid_count = 0
        
        # Process positive samples
        for img_path, ann_path in pairs:
            try:
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Failed to read image: {img_path}")
                    continue
                
                with open(ann_path, 'r') as f:
                    lines = f.readlines()
                
                valid_annotations = []
                for line in lines:
                    formatted_line = self._validate_keypoints(line)
                    if formatted_line:
                        valid_annotations.append(formatted_line)
                    else:
                        invalid_count += 1
                
                if valid_annotations:
                    samples.append({
                        'image_path': img_path,
                        'yolo_format': valid_annotations,
                        'is_negative': False
                    })
                else:
                    print(f"No valid annotations found in {ann_path}")
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Add negative samples
        samples.extend(negative_samples)
        
        print(f"Processed total {len(samples)} samples:")
        print(f"- Valid positive samples: {len(samples) - len(negative_samples)}")
        print(f"- Negative samples: {len(negative_samples)}")
        print(f"- Invalid annotations: {invalid_count}")
        
        return samples

    def prepare_dataset(self):
        """Splits the dataset into train, val, and test sets."""
        all_samples = self._prepare_samples()
        
        if not all_samples:
            raise ValueError("No valid samples found in the dataset.")
        
        train_ratio = self.config['training']['train_ratio']
        val_ratio = self.config['training']['val_ratio']
        
        train_data, temp = train_test_split(all_samples, train_size=train_ratio, 
                                          stratify=[s['is_negative'] for s in all_samples])
        val_data, test_data = train_test_split(temp, train_size=val_ratio/(1-train_ratio),
                                             stratify=[s['is_negative'] for s in temp])
        
        return train_data, val_data, test_data

    def prepare_yolo_dataset(self):
        """Prepares the dataset in YOLO format."""
        train_data, val_data, test_data = self.prepare_dataset()
        splits = {'train': train_data, 'val': val_data, 'test': test_data}
        
        total_processed = 0
        total_negative = 0
        
        for split_name, split_data in splits.items():
            print(f"\nProcessing {split_name} split: {len(split_data)} samples")
            
            for idx, sample in enumerate(split_data):
                try:
                    # Copy and rename image
                    src_path = sample['image_path']
                    dst_name = f"{idx:06d}.jpg"
                    dst_path = os.path.join(self.dataset_dir, split_name, 'images', dst_name)
                    
                    image = cv2.imread(src_path)
                    if image is None:
                        print(f"Failed to read image: {src_path}")
                        continue
                    
                    cv2.imwrite(dst_path, image)
                    
                    # Save annotations (empty file for negative samples)
                    label_path = os.path.join(self.dataset_dir, split_name, 'labels', f"{idx:06d}.txt")
                    if sample['is_negative']:
                        # Create empty label file for negative sample
                        open(label_path, 'w').close()
                        total_negative += 1
                    else:
                        # Write annotations for positive sample
                        with open(label_path, 'w') as f:
                            for line in sample['yolo_format']:
                                f.write(line + '\n')
                    
                    total_processed += 1
                    
                except Exception as e:
                    print(f"Error processing {src_path}: {e}")
                    continue
        
        print(f"\nTotal processed samples: {total_processed}")
        print(f"- Positive samples: {total_processed - total_negative}")
        print(f"- Negative samples: {total_negative}")
        
        # Create data.yaml
        data_yaml = {
            'path': self.dataset_dir,
            'train': os.path.join('train', 'images'),
            'val': os.path.join('val', 'images'),
            'test': os.path.join('test', 'images'),
            'nc': 1,
            'names': ['hand'],
            'kpt_shape': [1, 3],  # [number of keypoints, dimensions per keypoint]
            'task': 'pose'
        }
        
        yaml_path = os.path.join(self.dataset_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"\nDataset configuration saved to: {yaml_path}")
        return yaml_path