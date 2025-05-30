from typing import List, Dict, Optional
import os
import uuid
import time
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from ultralytics import YOLO
import numpy as np
import cv2
import logging
from PIL import Image
import io
import requests


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model for Mask and Hairnet detection using YOLOv8
    """
    
    def setup(self):
        """Configure model parameters and load the YOLO model
        """
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model configuration - use set() method instead of direct assignment
        self._model_version = "1.0.0"  # Store as an instance variable
        self.set("model_version", self._model_version)  # Set in Label Studio
        
        # Model path - update this to your model's location
        model_path = os.path.join(os.path.dirname(__file__), 'weights', 'best.pt')
        
        # Check if model exists, otherwise use the default YOLO model
        if not os.path.exists(model_path):
            self.logger.warning(f"Model not found at {model_path}, using default YOLOv8n")
            model_path = "yolov8n.pt"
        
        # Create results directory if it doesn't exist (like in test_single_image)
        self.results_dir = os.path.join(os.path.dirname(__file__), 'debug_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load model
        self.logger.info(f"Loading model from {model_path}")
        try:
            self.model = YOLO(model_path)
            # IMPORTANT: This line was commented out but is needed
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.model = None

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """Process images and return mask and hairnet detections
        """
        if self.model is None:
            return ModelResponse(predictions=[])
        
        predictions = []
        
        for task in tasks:
            self.logger.info(f"Processing task {task.get('id', 'unknown')}")
            
            # Check if image data is available in the task
            if 'data' not in task or not any(k for k in task['data'].keys() if k.endswith('image') or k.endswith('img')):
                continue
            
            # Find image URL
            image_url = None
            for key, value in task['data'].items():
                if key.endswith('image') or key.endswith('img'):
                    image_url = value
                    break
            
            if not image_url:
                continue
                
            # Get image file
            try:
                if image_url.startswith('http'):
                    image = self._get_image_from_url(image_url)
                else:
                    image_path = self.get_local_path(image_url, task_id=task['id'])
                    
                    # Use cv2 directly like in the test script instead of PIL
                    image_np = cv2.imread(image_path)
                    
                    # For debugging, save a copy of the input image
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    debug_image_path = os.path.join(self.results_dir, f"input_{timestamp}.jpg")
                    cv2.imwrite(debug_image_path, image_np)
                    self.logger.info(f"Saved input image to {debug_image_path}")
                
                # Run prediction with the model - IMPORTANT: Change conf_thres to conf
                self.logger.info(f"Running inference on image (shape: {image_np.shape})...")
                results = self.model(image_np, conf=0.25)  # Changed from conf_thres to conf
                result = results[0]  # Get first result
                
                self.logger.info(f"Found {len(result.boxes)} objects in the image")
                
                # Debug: Draw bounding boxes on image and save it
                img_debug = image_np.copy()
                for box in result.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get class and confidence
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Get class name (use result.names like in test script)
                    cls_name = result.names[cls_id]
                    
                    # Draw rectangle
                    cv2.rectangle(img_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(img_debug, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    self.logger.info(f"Detected {cls_name} with confidence {conf:.2f} at position {(x1, y1, x2, y2)}")
                
                # Save debug image
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                debug_output_path = os.path.join(self.results_dir, f"output_{timestamp}.jpg")
                cv2.imwrite(debug_output_path, img_debug)
                self.logger.info(f"Saved debug output to {debug_output_path}")
                
                # Extract detections
                detection_result = self._create_detection_result(result, task)
                if detection_result:
                    predictions.append({
                        "model_version": self._model_version,
                        "result": detection_result
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing image {image_url}: {str(e)}")
                self.logger.exception("Exception details:")
                continue
        
        return ModelResponse(predictions=predictions)
    
    def _get_image_from_url(self, url):
        """Download image from URL"""
        try:
            response = requests.get(url)
            image_data = np.asarray(bytearray(response.content), dtype="uint8")
            # Use cv2 to decode image like in test script
            return cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        except Exception as e:
            self.logger.error(f"Error downloading image from {url}: {str(e)}")
            return None
    
    def _create_detection_result(self, result, task):
        """Convert YOLO result to Label Studio format"""
        if not hasattr(result, 'boxes'):
            return []
            
        annotations = []
        boxes = result.boxes
        
        # Get image dimensions for relative coordinates calculation
        img_width = result.orig_shape[1]
        img_height = result.orig_shape[0]
        
        # Find image field name in task data
        image_field = None
        for key in task['data'].keys():
            if key.endswith('image') or key.endswith('img'):
                image_field = key
                break
                
        if not image_field:
            return []
            
        # Process each detection
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()  # Get box in [x1, y1, x2, y2] format
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Use result.names instead of self.labels
            cls_name = result.names[cls]
            
            # Convert to relative coordinates for Label Studio
            x_min, y_min, x_max, y_max = xyxy
            x = x_min / img_width * 100
            y = y_min / img_height * 100
            width = (x_max - x_min) / img_width * 100
            height = (y_max - y_min) / img_height * 100
            
            # Create annotation in Label Studio format
            annotations.append({
                "id": str(uuid.uuid4())[:10],
                "from_name": "label",  # Update this to match your Label Studio config
                "to_name": image_field,
                "type": "rectanglelabels",
                "score": conf,
                "value": {
                    "rectanglelabels": [cls_name],
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                }
            })
            
        return annotations
    
    def get_local_path(self, url, task_id=None):
        """Get local path for uploaded files in Label Studio.
        
        Args:
            url: The file URL or path
            task_id: Optional task ID
        
        Returns:
            Local file path that can be accessed by the ML backend
        """
        self.logger.info(f"Processing URL: {url}")
        
        # Case 1: URL starts with /data/upload - uploaded files path
        if url.startswith('/data/upload/'):
            # Convert to the path as it appears in our container's mount
            converted_path = url.replace('/data/upload/', '/label-studio/data/upload/')
            self.logger.info(f"Converted path: {converted_path}")
            return converted_path
            
        # Case 2: Full URL to Label Studio instance
        if url.startswith(('http://', 'https://')):
            # Use parent class method to handle full URLs
            return super().get_local_path(url, task_id)
        
        # Default case: return the URL as-is
        return url
    
    def fit(self, event, data, **kwargs):
        """Train or fine-tune the YOLO model on new annotations
        
        Args:
            event: Event type (usually "ANNOTATION_CREATED" or "ANNOTATION_UPDATED")
            data: List of annotations with task data
            **kwargs: Additional arguments
        
        Returns:
            Dict with training results
        """
        self.logger.info(f"Received {event} event with {len(data)} annotations")
        
        # Create a directory for training data
        train_dir = os.path.join(os.path.dirname(__file__), 'train_data')
        os.makedirs(train_dir, exist_ok=True)
        
        # Create images and labels directories for YOLO format
        images_dir = os.path.join(train_dir, 'images')
        labels_dir = os.path.join(train_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Process annotations and prepare training data
        prepared_count = 0
        for item in data:
            try:
                # Extract task data and annotations
                if 'annotations' not in item or not item['annotations']:
                    continue
                    
                annotations = item['annotations'][0]['result'] 
                task = item['task']
                
                # Find image URL
                image_url = None
                image_field = None
                for key, value in task['data'].items():
                    if key.endswith('image') or key.endswith('img'):
                        image_url = value
                        image_field = key
                        break
                
                if not image_url:
                    continue
                    
                # Get image file
                if image_url.startswith('http'):
                    image_np = self._get_image_from_url(image_url)
                    if image_np is None:
                        continue
                else:
                    image_path = self.get_local_path(image_url, task_id=task['id'])
                    image_np = cv2.imread(image_path)
                    if image_np is None:
                        self.logger.error(f"Could not read image from {image_path}")
                        continue
                
                # Save image for training
                image_filename = f"image_{task['id']}.jpg"
                image_save_path = os.path.join(images_dir, image_filename)
                cv2.imwrite(image_save_path, image_np)
                
                # Create YOLO format labels
                height, width = image_np.shape[:2]
                label_content = []
                
                # Process each annotation
                for ann in annotations:
                    if ann['type'] != 'rectanglelabels':
                        continue
                        
                    # Get label
                    label = ann['value']['rectanglelabels'][0]
                    
                    # Map label to class ID - using the exact mapping from your YAML file
                    class_mapping = {
                        'hairnet_improper': 0,
                        'hairnet_proper': 1,
                        'mask_improper': 2,
                        'mask_proper': 3
                    }
                    
                    if label not in class_mapping:
                        self.logger.warning(f"Unknown label: {label}, skipping")
                        continue
                        
                    class_id = class_mapping[label]
                    
                    # Get normalized coordinates (YOLO format: center_x, center_y, width, height)
                    # Label Studio uses % values (0-100), YOLO uses normalized (0-1)
                    x = ann['value']['x'] / 100.0
                    y = ann['value']['y'] / 100.0
                    w = ann['value']['width'] / 100.0
                    h = ann['value']['height'] / 100.0
                    
                    # Convert from top-left to center coordinates
                    center_x = x + w/2
                    center_y = y + h/2
                    
                    # Add to label file
                    label_content.append(f"{class_id} {center_x} {center_y} {w} {h}")
                
                # Save label file
                label_filename = f"image_{task['id']}.txt"
                label_save_path = os.path.join(labels_dir, label_filename)
                with open(label_save_path, 'w') as f:
                    f.write('\n'.join(label_content))
                    
                prepared_count += 1
                self.logger.info(f"Prepared training sample from task {task['id']}")
                    
            except Exception as e:
                self.logger.error(f"Error processing annotation: {str(e)}")
                self.logger.exception("Exception details:")
                continue
        
        self.logger.info(f"Prepared {prepared_count} samples for training")
        
        if prepared_count == 0:
            self.logger.warning("No valid samples prepared for training, aborting")
            return {"status": "error", "message": "No valid samples prepared for training"}
        
        # Create YAML config for training - using your exact YAML structure
        yaml_path = os.path.join(train_dir, 'dataset.yaml')
        yaml_content = f"""# Dataset configuration for YOLO training
    path: {train_dir}  # Dataset root directory
    train: images/
    val: images/  # Using same images for validation
    
    # Class names - matching your original training
    names:
      0: 'hairnet_improper'
      1: 'hairnet_proper'
      2: 'mask_improper'
      3: 'mask_proper'
    """
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        # Start training in a separate process
        try:
            # Define output directory
            output_dir = os.path.join(os.path.dirname(__file__), 'training_results')
            os.makedirs(output_dir, exist_ok=True)
            
            # Run training in background thread to avoid blocking
            import threading
            
            def train_model():
                try:
                    # Use the current model as starting point
                    model_path = os.path.join(os.path.dirname(__file__), 'weights', 'best.pt')
                    
                    # Command to fine-tune the model with smaller epochs and batch size for faster results
                    command = f"yolo train model={model_path} data={yaml_path} epochs=5 imgsz=640 batch=4 project={output_dir}"
                    
                    self.logger.info(f"Starting training with command: {command}")
                    import subprocess
                    process = subprocess.Popen(command.split(), 
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    
                    # Log training output
                    self.logger.info(f"Training completed with exit code {process.returncode}")
                    
                    if process.returncode != 0:
                        self.logger.error(f"Training error: {stderr.decode()}")
                        return
                    
                    # Copy trained model to weights folder
                    new_weights = os.path.join(output_dir, 'train', 'weights', 'best.pt')
                    if os.path.exists(new_weights):
                        import shutil
                        model_path = os.path.join(os.path.dirname(__file__), 'weights', 'best.pt')
                        # Backup old model
                        if os.path.exists(model_path):
                            backup_path = os.path.join(os.path.dirname(__file__), 'weights', f'best_backup_{int(time.time())}.pt')
                            shutil.copy(model_path, backup_path)
                            self.logger.info(f"Created backup of old model at {backup_path}")
                            
                        # Copy new model
                        shutil.copy(new_weights, model_path)
                        self.logger.info(f"Updated model weights at {model_path}")
                        
                        # Reload model
                        self.model = YOLO(model_path)
                        self.logger.info("Reloaded model with new weights")
                    else:
                        self.logger.error(f"Training completed but no weights file found at {new_weights}")
                    
                except Exception as e:
                    self.logger.error(f"Error during training: {str(e)}")
                    self.logger.exception("Training exception details:")
            
            # Start training thread
            training_thread = threading.Thread(target=train_model)
            training_thread.daemon = True
            training_thread.start()
            
            # Update model version
            version_parts = self._model_version.split('.')
            new_version = f"{version_parts[0]}.{version_parts[1]}.{int(version_parts[2]) + 1}"
            self._model_version = new_version
            self.set("model_version", self._model_version)
            
            self.logger.info(f"Updated model version to {self._model_version}")
            return {"status": "ok", "message": f"Training started in background with {prepared_count} images"}
            
        except Exception as e:
            self.logger.error(f"Error starting training: {str(e)}")
            self.logger.exception("Exception details:")
            return {"status": "error", "message": str(e)}