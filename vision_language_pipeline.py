import os
import torch
import clip
import numpy as np
import cv2
from PIL import Image
from object_detection import ObjectDetector

class VisionLanguagePipeline:
    def __init__(self, clip_model="ViT-B/32"):
        """Initialize vision-language pipeline with CLIP
        
        Args:
            clip_model: CLIP model variant to use
        """
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.clip_model, self.preprocess = clip.load(clip_model, device=self.device)
        
        # Initialize object detector
        self.detector = ObjectDetector()
        
        # Common templates for zero-shot classification
        self.templates = [
            "a photo of a {}.",
            "a close-up photo of a {}.",
            "a photo of a small {}.",
            "a photo of a large {}.",
            "a photo of the {}.",
        ]
        
    def detect_objects(self, image_path, conf_threshold=0.25):
        """Detect objects in an image using YOLOv5
        
        Args:
            image_path: Path to the image file
            conf_threshold: Confidence threshold
            
        Returns:
            detections: List of detection dictionaries
            annotated_image: Image with bounding boxes
        """
        return self.detector.detect(image_path, conf_threshold)
    
    def classify_with_clip(self, image, candidate_labels):
        """Classify image with CLIP using candidate labels
        
        Args:
            image: PIL Image, numpy array, or path to image
            candidate_labels: List of text labels to choose from
            
        Returns:
            predictions: Dictionary mapping labels to probabilities
        """
        # Handle different input types
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Prepare image for CLIP
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Prepare text prompts
        text_prompts = []
        for label in candidate_labels:
            text_prompts.extend([template.format(label) for template in self.templates])
        
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        # Compute features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_tokens)
            
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Aggregate template results
        num_templates = len(self.templates)
        predictions = {}
        
        for i, label in enumerate(candidate_labels):
            template_similarities = similarity[0, i*num_templates:(i+1)*num_templates]
            predictions[label] = template_similarities.mean().item()
        
        # Sort by probability
        sorted_predictions = {k: v for k, v in sorted(predictions.items(), 
                                                     key=lambda item: item[1], 
                                                     reverse=True)}
        
        return sorted_predictions
    
    def answer_question(self, image, question, candidate_answers=None):
        """Answer a question about an image
        
        Args:
            image: PIL Image, numpy array, or path to image
            question: Question string
            candidate_answers: Optional list of possible answers
            
        Returns:
            answer: Most likely answer
            confidence: Confidence score
        """
        # Handle different input types
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        # Prepare image for CLIP
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate candidate answers if not provided
        if candidate_answers is None:
            # Default answers for common question types
            if "what" in question.lower():
                # Detect objects first
                img_np = np.array(image)
                detections, _ = self.detector.detect(img_np)
                # Use detected objects as candidate answers
                candidate_answers = [det["class_name"] for det in detections]
                # Add some generic answers
                candidate_answers.extend(["person", "animal", "vehicle", "furniture", "appliance", "food"])
                # Remove duplicates
                candidate_answers = list(set(candidate_answers))
            elif "where" in question.lower():
                candidate_answers = ["inside", "outside", "on top", "under", "beside", "in front of", "behind"]
            elif "how many" in question.lower():
                candidate_answers = ["0", "1", "2", "3", "4", "5", "many"]
            elif "color" in question.lower() or "colour" in question.lower():
                candidate_answers = ["red", "green", "blue", "yellow", "white", "black", "orange", "purple", "brown", "pink", "gray"]
            else:
                candidate_answers = ["yes", "no", "maybe"]
        
        # Format prompts with the question
        text_prompts = [f"Question: {question} Answer: {answer}." for answer in candidate_answers]
        
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        # Compute features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_tokens)
            
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Get top prediction
        values, indices = similarity[0].topk(len(candidate_answers))
        
        top_answer = candidate_answers[indices[0].item()]
        top_confidence = values[0].item()
        
        return top_answer, top_confidence, dict(zip(candidate_answers, values.tolist()))
    
    def generate_scene_description(self, image_path):
        """Generate a basic scene description
        
        Args:
            image_path: Path to the image or image as numpy array
            
        Returns:
            description: String describing the scene
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image_path, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
        else:
            image = image_path
            
        # Get object detections
        detections, _ = self.detector.detect(image)
        
        # Count objects by class
        object_counts = {}
        for det in detections:
            class_name = det['class_name']
            if class_name in object_counts:
                object_counts[class_name] += 1
            else:
                object_counts[class_name] = 1
        
        # Generate scene description
        if not object_counts:
            return "I don't see any recognizable objects in this image."
        
        # Generate description based on counts
        description_parts = []
        for obj, count in object_counts.items():
            if count == 1:
                description_parts.append(f"a {obj}")
            else:
                description_parts.append(f"{count} {obj}s")
        
        # Use CLIP for scene classification
        scene_categories = [
            "indoor scene", "outdoor scene", "natural landscape", 
            "urban environment", "kitchen", "living room", "bedroom", 
            "office", "street", "park", "beach", "forest", "mountain"
        ]
        
        scene_predictions = self.classify_with_clip(image, scene_categories)
        top_scene = next(iter(scene_predictions))
        
        # Combine into a coherent description
        if len(description_parts) == 1:
            objects_text = f"There is {description_parts[0]}"
        elif len(description_parts) == 2:
            objects_text = f"There are {description_parts[0]} and {description_parts[1]}"
        else:
            objects_text = "There are " + ", ".join(description_parts[:-1]) + f", and {description_parts[-1]}"
        
        description = f"This appears to be an {top_scene}. {objects_text} in the image."
        
        return description

# Example usage
if __name__ == "__main__":
    pipeline = VisionLanguagePipeline()
    
    # Replace with your image path
    image_path = "example_kitchen.jpg"
    
    # Detect objects
    detections, annotated_image = pipeline.detect_objects(image_path)
    print(f"Detected {len(detections)} objects:")
    for det in detections:
        print(f"- {det['class_name']} ({det['confidence']:.2f})")
    
    # Answer a question
    question = "What is in the image?"
    answer, confidence, all_answers = pipeline.answer_question(image_path, question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer} (confidence: {confidence:.2f})")
    
    # Generate scene description
    description = pipeline.generate_scene_description(image_path)
    print(f"\nScene Description: {description}")