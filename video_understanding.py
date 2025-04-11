import os
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F
from object_detection import ObjectDetector

class VideoUnderstandingModule:
    def __init__(self, action_model="slowfast_r50"):
        """Initialize video understanding module
        
        Args:
            action_model: Action recognition model to use
                Options: "slowfast_r50", "x3d_m", etc.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # We'll use CLIP for action recognition instead of SlowFast
        self.action_model = None
        self.kinetics_labels = self._load_kinetics_labels()
            
        # Initialize object detector
        self.object_detector = ObjectDetector()
    
    def _load_kinetics_labels(self):
        """Load Kinetics-400 class labels"""
        # First try to download the labels file
        try:
            import requests
            url = "https://raw.githubusercontent.com/facebookresearch/pytorchvideo/main/pytorchvideo/data/kinetics.py"
            response = requests.get(url)
            
            labels = []
            for line in response.text.split('\n'):
                if '"' in line and ':' in line:
                    label = line.split('"')[1]
                    labels.append(label)
            
            if len(labels) == 0:
                # Fallback to hard-coded labels (first few)
                labels = ["abseiling", "air drumming", "answering questions", "applauding", 
                          "applying cream", "archery", "arm wrestling", "arranging flowers", 
                          "assembling computer", "auctioning", "baby waking up", "baking cookies",
                          "balloon blowing", "bandaging", "barbequing", "bartending"]
                
            return labels
            
        except Exception as e:
            print(f"Error loading Kinetics labels: {e}")
            return ["action_" + str(i) for i in range(400)]  # Fallback
    
    def extract_keyframes(self, video_path, method='uniform', n_frames=5, threshold=40.0):
        """Extract key frames from a video
        
        Args:
            video_path: Path to the video file
            method: Frame extraction method ('uniform', 'change')
            n_frames: Number of frames to extract (for 'uniform')
            threshold: Difference threshold (for 'change')
            
        Returns:
            frames: List of extracted frames (as numpy arrays)
            frame_indices: List of frame indices in the video
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0:
            raise ValueError(f"Could not read frames from {video_path}")
        
        frames = []
        frame_indices = []
        
        if method == 'uniform':
            # Extract frames at regular intervals
            indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    frame_indices.append(idx)
        
        elif method == 'change':
            # Extract frames based on visual change
            prev_frame = None
            prev_gray = None
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    # Calculate difference with previous frame
                    diff = cv2.absdiff(gray, prev_gray)
                    diff_mean = np.mean(diff)
                    
                    if diff_mean > threshold:
                        frames.append(frame)
                        frame_indices.append(frame_idx)
                elif frame_idx == 0:  # Always include the first frame
                    frames.append(frame)
                    frame_indices.append(frame_idx)
                
                prev_gray = gray
                frame_idx += 1
                
                # Limit the number of frames
                if len(frames) >= n_frames:
                    break
        
        cap.release()
        
        return frames, frame_indices
    
    def recognize_actions(self, video_path, clip_duration=2.0, stride=1.0):
        """Recognize actions in video clips
        
        Args:
            video_path: Path to the video file
            clip_duration: Duration of each clip in seconds
            stride: Stride between clips in seconds
            
        Returns:
            actions: List of dictionaries with actions and timestamps
        """
        if self.action_model is None:
            return [{"action": "unknown", "confidence": 0.0, "start_time": 0.0, "end_time": 0.0}]
        
        # Let's use a simpler approach with keyframes since SlowFast requires complex input formatting
        # Extract keyframes first
        frames, frame_indices = self.extract_keyframes(video_path, method='uniform', n_frames=10)
        
        # Get video information
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        cap.release()
        
        # Create a simpler action recognition approach using CLIP
        # We'll classify each keyframe with action-related categories
        action_categories = [
            "walking", "running", "jumping", "sitting", "standing",
            "cooking", "eating", "drinking", "talking", "reading",
            "writing", "typing", "opening", "closing", "picking up",
            "putting down", "throwing", "catching", "pushing", "pulling",
            "lifting", "carrying", "cleaning", "washing", "driving"
        ]
        
        actions = []
        
        # Import CLIP if needed
        try:
            import clip
            from PIL import Image
            import torch
            
            # Load CLIP model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            
            # Process each keyframe
            for i, frame in enumerate(frames):
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Preprocess
                image_input = preprocess(pil_image).unsqueeze(0).to(device)
                
                # Prepare text prompts
                text_prompts = [f"a person {action}" for action in action_categories]
                text_tokens = clip.tokenize(text_prompts).to(device)
                
                # Get predictions
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    text_features = model.encode_text(text_tokens)
                
                    # Normalize features
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                
                    # Compute similarity
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # Get top prediction
                values, indices = similarity[0].topk(1)
                action_idx = indices[0].item()
                confidence = values[0].item()
                
                # Calculate timestamp
                timestamp = frame_indices[i] / fps
                
                # Add to actions
                actions.append({
                    "action": action_categories[action_idx],
                    "confidence": float(confidence),
                    "start_time": max(0, timestamp - 0.5),
                    "end_time": min(video_duration, timestamp + 0.5)
                })
                
        except Exception as e:
            print(f"Error in action recognition: {e}")
            actions = [{"action": "unknown", "confidence": 0.0, "start_time": 0.0, "end_time": 0.0}]
        
        return actions
    
    def identify_critical_objects(self, frames):
        """Identify critical objects in key frames
        
        Args:
            frames: List of video frames
            
        Returns:
            critical_objects: Dictionary mapping object names to occurrence counts
        """
        all_detections = []
        
        # Detect objects in each frame
        for frame in frames:
            detections, _ = self.object_detector.detect(frame)
            all_detections.extend(detections)
        
        # Count object occurrences
        object_counts = {}
        for det in all_detections:
            obj_name = det['class_name']
            if obj_name in object_counts:
                object_counts[obj_name] += 1
            else:
                object_counts[obj_name] = 1
        
        # Sort by occurrence count (most frequent first)
        sorted_objects = {k: v for k, v in sorted(object_counts.items(), 
                                                  key=lambda item: item[1], 
                                                  reverse=True)}
        
        return sorted_objects
    
    def analyze_video(self, video_path, n_keyframes=8):
        """Analyze a video to extract actions, key frames, and objects
        
        Args:
            video_path: Path to the video file
            n_keyframes: Number of key frames to extract
            
        Returns:
            analysis: Dictionary with analysis results
        """
        # Extract key frames
        print("Extracting key frames...")
        keyframes, frame_indices = self.extract_keyframes(video_path, 
                                                         method='change', 
                                                         n_frames=n_keyframes)
        
        # Recognize actions
        print("Recognizing actions...")
        actions = self.recognize_actions(video_path)
        
        # Identify critical objects
        print("Identifying critical objects...")
        critical_objects = self.identify_critical_objects(keyframes)
        
        # Calculate frame timestamps
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        frame_timestamps = [idx / fps for idx in frame_indices]
        
        # Create analysis result
        analysis = {
            "video_path": video_path,
            "keyframes": {
                "frames": keyframes,
                "indices": frame_indices,
                "timestamps": frame_timestamps
            },
            "actions": actions,
            "critical_objects": critical_objects
        }
        
        return analysis
    
    def summarize_video(self, analysis):
        """Generate a summary of the video analysis
        
        Args:
            analysis: Analysis dictionary from analyze_video()
            
        Returns:
            summary: Text summary of the video
        """
        # Extract key information
        actions = analysis["actions"]
        critical_objects = analysis["critical_objects"]
        
        # Generate action summary
        action_summary = []
        for action in actions:
            start_time = f"{action['start_time']:.1f}s"
            end_time = f"{action['end_time']:.1f}s"
            action_text = f"{action['action']} ({start_time} - {end_time})"
            action_summary.append(action_text)
        
        # Generate object summary
        object_summary = []
        for obj, count in critical_objects.items():
            object_summary.append(f"{obj} (x{count})")
        
        # Combine into overall summary
        summary = "Video Summary:\n\n"
        
        # Actions section
        summary += "Actions:\n"
        if action_summary:
            for i, action in enumerate(action_summary[:5], 1):  # Top 5 actions
                summary += f"{i}. {action}\n"
            if len(action_summary) > 5:
                summary += f"... and {len(action_summary) - 5} more actions\n"
        else:
            summary += "No actions detected\n"
        
        summary += "\n"
        
        # Objects section
        summary += "Key Objects:\n"
        if object_summary:
            for i, obj in enumerate(object_summary[:10], 1):  # Top 10 objects
                summary += f"{i}. {obj}\n"
            if len(object_summary) > 10:
                summary += f"... and {len(object_summary) - 10} more objects\n"
        else:
            summary += "No objects detected\n"
        
        return summary

# Example usage
if __name__ == "__main__":
    video_module = VideoUnderstandingModule()
    
    # Replace with your video path
    video_path = "example_video.mp4"
    
    # Analyze video
    analysis = video_module.analyze_video(video_path)
    
    # Generate summary
    summary = video_module.summarize_video(analysis)
    print(summary)
    
    # Display key frames
    import matplotlib.pyplot as plt
    
    # Plot key frames
    keyframes = analysis["keyframes"]["frames"]
    timestamps = analysis["keyframes"]["timestamps"]
    
    fig, axs = plt.subplots(2, 4, figsize=(15, 8))
    axs = axs.flatten()
    
    for i, (frame, ts) in enumerate(zip(keyframes, timestamps)):
        if i >= len(axs):
            break
            
        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        axs[i].imshow(frame_rgb)
        axs[i].set_title(f"Frame {i+1} - {ts:.2f}s")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()