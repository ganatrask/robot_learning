import os
import cv2
import numpy as np
import argparse
from vision_language_pipeline import VisionLanguagePipeline
from video_understanding import VideoUnderstandingModule

class RobotVisionLearningSystem:
    def __init__(self):
        """Initialize the robot vision learning system"""
        print("Initializing Vision-Language Pipeline...")
        self.vlm = VisionLanguagePipeline()
        
        print("Initializing Video Understanding Module...")
        self.video_module = VideoUnderstandingModule()
        
        print("System initialized and ready!")
    
    def process_image(self, image_path):
        """Process a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            results: Dictionary with analysis results
        """
        print(f"Processing image: {image_path}")
        
        # Detect objects
        detections, annotated_image = self.vlm.detect_objects(image_path)
        
        # Generate scene description
        description = self.vlm.generate_scene_description(image_path)
        
        # Answer some basic questions about the image
        questions = [
            "What objects are in the scene?",
            "Is there a person in the image?",
            "What is the main object in this image?"
        ]
        
        qa_results = {}
        for question in questions:
            answer, confidence, all_answers = self.vlm.answer_question(image_path, question)
            qa_results[question] = {
                "answer": answer,
                "confidence": confidence,
                "all_answers": all_answers
            }
        
        # Compile results
        results = {
            "image_path": image_path,
            "detections": detections,
            "annotated_image": annotated_image,
            "description": description,
            "qa_results": qa_results
        }
        
        return results
    
    def process_video(self, video_path):
        """Process a video to extract actions, keyframes, and objects
        
        Args:
            video_path: Path to the video file
            
        Returns:
            analysis: Dictionary with analysis results
        """
        print(f"Processing video: {video_path}")
        
        # Use video module to analyze the video
        analysis = self.video_module.analyze_video(video_path)
        
        # Generate a summary
        summary = self.video_module.summarize_video(analysis)
        analysis["summary"] = summary
        
        # Process key frames with VLM
        keyframe_descriptions = []
        for i, frame in enumerate(analysis["keyframes"]["frames"]):
            # Generate description for each keyframe
            description = self.vlm.generate_scene_description(frame)
            keyframe_descriptions.append(description)
        
        analysis["keyframe_descriptions"] = keyframe_descriptions
        
        return analysis
    
    def save_results(self, results, output_dir):
        """Save analysis results to output directory
        
        Args:
            results: Analysis results (from process_image or process_video)
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine if results are from image or video processing
        if "detections" in results:  # Image results
            # Save annotated image
            image_path = os.path.join(output_dir, "annotated_image.jpg")
            cv2.imwrite(image_path, results["annotated_image"])
            
            # Save text results
            text_path = os.path.join(output_dir, "analysis.txt")
            with open(text_path, "w") as f:
                f.write(f"Scene Description: {results['description']}\n\n")
                
                f.write("Detected Objects:\n")
                for i, det in enumerate(results["detections"]):
                    f.write(f"{i+1}. {det['class_name']} "
                            f"(confidence: {det['confidence']:.2f})\n")
                
                f.write("\nQuestion-Answering Results:\n")
                for question, result in results["qa_results"].items():
                    f.write(f"Q: {question}\n")
                    f.write(f"A: {result['answer']} "
                            f"(confidence: {result['confidence']:.2f})\n\n")
            
            print(f"Image analysis saved to {output_dir}")
            
        else:  # Video results
            # Save keyframes
            keyframes_dir = os.path.join(output_dir, "keyframes")
            os.makedirs(keyframes_dir, exist_ok=True)
            
            for i, frame in enumerate(results["keyframes"]["frames"]):
                frame_path = os.path.join(keyframes_dir, f"keyframe_{i+1}.jpg")
                cv2.imwrite(frame_path, frame)
            
            # Save summary
            summary_path = os.path.join(output_dir, "video_summary.txt")
            with open(summary_path, "w") as f:
                f.write(results["summary"])
                
                f.write("\n\nKeyframe Descriptions:\n")
                for i, desc in enumerate(results["keyframe_descriptions"]):
                    timestamp = results["keyframes"]["timestamps"][i]
                    f.write(f"\nKeyframe {i+1} ({timestamp:.2f}s):\n")
                    f.write(f"{desc}\n")
            
            print(f"Video analysis saved to {output_dir}")


def main():
    """Command line interface for the Robot Vision Learning System"""
    parser = argparse.ArgumentParser(
        description="Robot Vision Learning System for processing images and videos"
    )
    
    parser.add_argument("--input", "-i", required=True, 
                        help="Path to input image or video file")
    parser.add_argument("--output", "-o", default="./output",
                       help="Directory to save output results (default: ./output)")
    parser.add_argument("--mode", "-m", choices=["image", "video", "auto"],
                       default="auto", help="Processing mode (default: auto)")
    
    args = parser.parse_args()
    
    # Determine file type if mode is auto
    if args.mode == "auto":
        if args.input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            args.mode = "image"
        elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            args.mode = "video"
        else:
            print(f"Could not determine file type for {args.input}. Please specify --mode.")
            return
    
    # Initialize system
    system = RobotVisionLearningSystem()
    
    # Process input
    if args.mode == "image":
        results = system.process_image(args.input)
    else:  # video mode
        results = system.process_video(args.input)
    
    # Save results
    system.save_results(results, args.output)


if __name__ == "__main__":
    main()

