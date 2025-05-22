from typing import List, Dict, Optional
import os
import uuid
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
        
        # Load model
        self.logger.info(f"Loading model from {model_path}")
        try:
            self.model = YOLO(model_path)
            self.labels = {0: 'Hairnet', 1: 'Mask'}
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
                    image = self.get_local_path(image_url, task_id=task['id'])
                    image = Image.open(image)
                    
                # Convert PIL image to numpy array for YOLO
                if isinstance(image, Image.Image):
                    image_np = np.array(image)
                else:
                    image_np = image
                    
                # Run prediction with the model
                results = self.model(image_np)
                result = results[0]  # Get first result
                
                # Extract detections
                detection_result = self._create_detection_result(result, task)
                if detection_result:
                    predictions.append({
                        "model_version": self._model_version,
                        "result": detection_result
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing image {image_url}: {str(e)}")
                continue
        
        return ModelResponse(predictions=predictions)
    
    def _get_image_from_url(self, url):
        """Download image from URL"""
        try:
            response = requests.get(url)
            return Image.open(io.BytesIO(response.content))
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
            label = self.labels.get(cls, f"class_{cls}")
            
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
                    "rectanglelabels": [label],
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                }
            })
            
        return annotations
    
    def fit(self, event, data, **kwargs):
        """
        This method could be used for fine-tuning your YOLO model with new annotations
        """
        self.logger.info(f"Received {event} event with {len(data)} annotations")
        
        # For now, just log the event and update model version
        version_parts = self._model_version.split('.')
        new_version = f"{version_parts[0]}.{version_parts[1]}.{int(version_parts[2]) + 1}"
        self._model_version = new_version
        self.set("model_version", self._model_version)
        
        self.logger.info(f"Updated model version to {self._model_version}")