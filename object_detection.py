import torch
import cv2
import numpy as np
from PIL import Image

class ObjectDetector:
    def __init__(self, model_size='s'):
        """Initialize YOLOv5 object detector
        
        Args:
            model_size: Size of YOLOv5 model ('n', 's', 'm', 'l', 'x')
        """
        self.model = torch.hub.load('ultralytics/yolov5', f'yolov5{model_size}')
        self.classes = self.model.names
    
    def detect(self, image_path, conf_threshold=0.25):
        """Detect objects in an image
        
        Args:
            image_path: Path to the image file, PIL Image, or numpy array
            conf_threshold: Confidence threshold for detections
            
        Returns:
            detections: List of dictionaries containing detection info
            annotated_image: Image with bounding boxes drawn
        """
        # Load image
        if isinstance(image_path, str):
            img = Image.open(image_path)
        elif isinstance(image_path, np.ndarray):
            # Convert numpy array to PIL Image
            img = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
        else:
            img = image_path  # Assume PIL Image
            
        # Run inference
        results = self.model(img)
        
        # Process results
        results.conf = conf_threshold  # Filter by confidence
        
        # Convert to list of dictionaries for easier handling
        detections = []
        for *box, conf, cls in results.xyxy[0]:  # xyxy, confidence, class
            x1, y1, x2, y2 = [int(val.item()) for val in box]
            class_id = int(cls.item())
            class_name = self.classes[class_id]
            confidence = float(conf.item())
            
            detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'box': (x1, y1, x2, y2)
            })
        
        # Get annotated image
        annotated_image = results.render()[0]
        
        return detections, annotated_image
    
    def get_object_crops(self, image_path, detections):
        """Extract cropped images of detected objects
        
        Args:
            image_path: Path to the image file
            detections: List of detection dictionaries
            
        Returns:
            crops: Dictionary mapping object IDs to cropped images
        """
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        else:
            img = np.array(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        crops = {}
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']
            crop = img[y1:y2, x1:x2]
            crops[f"{det['class_name']}_{i}"] = crop
            
        return crops

# Example usage
if __name__ == "__main__":
    detector = ObjectDetector()
    
    # Replace with your image path
    image_path = "example_image.jpg"
    
    detections, annotated_image = detector.detect(image_path)
    
    print(f"Detected {len(detections)} objects:")
    for det in detections:
        print(f"- {det['class_name']} ({det['confidence']:.2f})")
    
    # Convert to RGB for display
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Display with matplotlib
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()