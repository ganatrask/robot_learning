import os
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np
import json
import time
import datetime
import shutil

# Import our vision system components
from vision_language_pipeline import VisionLanguagePipeline
from video_understanding import VideoUnderstandingModule
from robot_vision_learning import RobotVisionLearningSystem

class RobotVisionGUI:
    def save_results(self):
        """Save the analysis results and media"""
        if not self.current_results or not self.current_media_path:
            messagebox.showinfo("Save Results", "No analysis results to save.")
            return
        
        # Ask for save directory
        save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
        if not save_dir:
            return
            
        # Create timestamp for unique folder naming
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        media_type = self.current_media_type
        
        # Create a directory for this save
        save_folder = os.path.join(save_dir, f"{media_type}_analysis_{timestamp}")
        os.makedirs(save_folder, exist_ok=True)
        
        try:
            # Save the media file (copy original)
            media_filename = os.path.basename(self.current_media_path)
            media_save_path = os.path.join(save_folder, media_filename)
            shutil.copy2(self.current_media_path, media_save_path)
            
            # Save the analysis results as text
            analysis_text_path = os.path.join(save_folder, "analysis_results.txt")
            with open(analysis_text_path, "w") as f:
                f.write(self.results_text.get(1.0, tk.END))
            
            # Save JSON data (for programmatic access later)
            json_data = self.prepare_json_data()
            json_path = os.path.join(save_folder, "analysis_data.json")
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)
            
            # For images, save the annotated image
            if self.current_media_type == 'image':
                annotated_path = os.path.join(save_folder, "annotated_" + media_filename)
                cv2.imwrite(annotated_path, self.current_results["annotated_image"])
            
            # For videos, save keyframes
            elif self.current_media_type == 'video':
                keyframes_dir = os.path.join(save_folder, "keyframes")
                os.makedirs(keyframes_dir, exist_ok=True)
                
                for i, frame in enumerate(self.current_results["keyframes"]["frames"]):
                    frame_path = os.path.join(keyframes_dir, f"keyframe_{i+1}.jpg")
                    cv2.imwrite(frame_path, frame)
            
            # Show success message
            messagebox.showinfo(
                "Save Complete", 
                f"Analysis results saved to:\n{save_folder}"
            )
            
        except Exception as e:
            messagebox.showerror(
                "Save Error", 
                f"Error saving results: {str(e)}"
            )
    
    def prepare_json_data(self):
        """Prepare analysis data for JSON saving"""
        json_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "media_type": self.current_media_type,
            "media_path": self.current_media_path
        }
        
        # Copy results but remove non-serializable data
        if self.current_media_type == 'image':
            # For images, we can save most data except the annotated image
            results_copy = dict(self.current_results)
            # Remove the numpy array which can't be serialized to JSON
            if "annotated_image" in results_copy:
                del results_copy["annotated_image"]
            json_data["analysis"] = results_copy
            
        elif self.current_media_type == 'video':
            # For videos, we need to handle keyframes and other non-serializable data
            results_copy = {}
            
            # Copy serializable data
            for key, value in self.current_results.items():
                if key != "keyframes":
                    results_copy[key] = value
            
            # Handle keyframes specially
            if "keyframes" in self.current_results:
                keyframes_data = {}
                if "timestamps" in self.current_results["keyframes"]:
                    keyframes_data["timestamps"] = self.current_results["keyframes"]["timestamps"]
                if "indices" in self.current_results["keyframes"]:
                    keyframes_data["indices"] = self.current_results["keyframes"]["indices"]
                results_copy["keyframes"] = keyframes_data
            
            json_data["analysis"] = results_copy
        
        return json_datadef __init__(self, root):
        self.root = root
        self.root.title("Robot Vision Learning System")
        self.root.geometry("1200x800")
        
        # Initialize the vision system (will happen in a separate thread)
        self.system = None
        self.system_ready = False
        
        # Track current media
        self.current_media_path = None
        self.current_media_type = None  # 'image' or 'video'
        self.current_results = None
        self.video_cap = None
        self.video_playing = False
        self.current_frame = None
        
        # Create GUI components
        self.create_widgets()
        
        # Start loading the vision system in a background thread
        self.loading_label.config(text="Loading vision system... Please wait.")
        thread = threading.Thread(target=self.load_vision_system)
        thread.daemon = True
        thread.start()
        
    def create_widgets(self):
        # Create main frames
        self.top_frame = ttk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        self.content_frame = ttk.Frame(self.root)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (for media display)
        self.left_panel = ttk.Frame(self.content_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel (for results and interaction)
        self.right_panel = ttk.Frame(self.content_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Top controls
        self.create_top_controls()
        
        # Media display area
        self.create_media_display()
        
        # Results and interactions area
        self.create_results_area()
        
    def create_top_controls(self):
        # Loading indicator
        self.loading_label = ttk.Label(self.top_frame, text="Welcome to Robot Vision System")
        self.loading_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Load Image button
        self.load_image_btn = ttk.Button(self.top_frame, text="Load Image", 
                                         command=self.load_image, state=tk.DISABLED)
        self.load_image_btn.pack(side=tk.LEFT, padx=5)
        
        # Load Video button
        self.load_video_btn = ttk.Button(self.top_frame, text="Load Video", 
                                         command=self.load_video, state=tk.DISABLED)
        self.load_video_btn.pack(side=tk.LEFT, padx=5)
        
        # Analyze button
        self.analyze_btn = ttk.Button(self.top_frame, text="Analyze", 
                                     command=self.analyze_media, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Save Results button
        self.save_btn = ttk.Button(self.top_frame, text="Save Results", 
                                  command=self.save_results, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
    def create_media_display(self):
        # Frame for display
        self.display_frame = ttk.LabelFrame(self.left_panel, text="Media")
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas for media display
        self.canvas = tk.Canvas(self.display_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Video controls (initially hidden)
        self.video_controls = ttk.Frame(self.left_panel)
        self.play_btn = ttk.Button(self.video_controls, text="Play", command=self.toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        self.video_slider = ttk.Scale(self.video_controls, from_=0, to=100, orient=tk.HORIZONTAL)
        self.video_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
    def create_results_area(self):
        # Notebook for results
        self.results_notebook = ttk.Notebook(self.right_panel)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Analysis tab
        self.analysis_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.analysis_frame, text="Analysis")
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(self.analysis_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Q&A tab
        self.qa_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.qa_frame, text="Question & Answer")
        
        # Question entry
        self.question_frame = ttk.Frame(self.qa_frame)
        self.question_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.question_label = ttk.Label(self.question_frame, text="Ask a question about the image:")
        self.question_label.pack(side=tk.LEFT, padx=5)
        
        self.question_entry = ttk.Entry(self.question_frame, width=40)
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.ask_btn = ttk.Button(self.question_frame, text="Ask", 
                                 command=self.ask_question, state=tk.DISABLED)
        self.ask_btn.pack(side=tk.LEFT, padx=5)
        
        # Answer display
        self.answer_text = scrolledtext.ScrolledText(self.qa_frame, height=10, wrap=tk.WORD)
        self.answer_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def load_vision_system(self):
        """Load the vision system in a background thread"""
        try:
            self.system = RobotVisionLearningSystem()
            self.system_ready = True
            
            # Update UI on the main thread
            self.root.after(0, self.system_loaded)
        except Exception as e:
            error_msg = f"Error loading vision system: {str(e)}"
            self.root.after(0, lambda: self.loading_label.config(
                text=error_msg, foreground="red"))
    
    def system_loaded(self):
        """Called when the vision system is loaded"""
        self.loading_label.config(text="Vision system ready!")
        self.load_image_btn.config(state=tk.NORMAL)
        self.load_video_btn.config(state=tk.NORMAL)
    
    def load_image(self):
        """Open file dialog to load an image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if file_path:
            self.current_media_path = file_path
            self.current_media_type = 'image'
            self.display_image(file_path)
            self.analyze_btn.config(state=tk.NORMAL)
            self.ask_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.DISABLED)
            self.results_text.delete(1.0, tk.END)
            self.answer_text.delete(1.0, tk.END)
            
            # Hide video controls
            self.video_controls.pack_forget()
            
            # Display file name in loading label
            self.loading_label.config(text=f"Loaded image: {os.path.basename(file_path)}")
    
    def load_video(self):
        """Open file dialog to load a video"""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm")]
        )
        
        if file_path:
            self.current_media_path = file_path
            self.current_media_type = 'video'
            
            # Open video and display first frame
            self.video_cap = cv2.VideoCapture(file_path)
            ret, frame = self.video_cap.read()
            
            if ret:
                self.current_frame = frame
                self.display_frame_image(frame)
                self.analyze_btn.config(state=tk.NORMAL)
                self.ask_btn.config(state=tk.DISABLED)
                self.save_btn.config(state=tk.DISABLED)
                self.results_text.delete(1.0, tk.END)
                self.answer_text.delete(1.0, tk.END)
                
                # Show video controls
                self.video_controls.pack(fill=tk.X, padx=5, pady=5)
                
                # Configure video slider
                total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_slider.config(to=total_frames - 1)
                self.video_slider.set(0)
                
                # Display file name in loading label
                self.loading_label.config(text=f"Loaded video: {os.path.basename(file_path)}")
            else:
                self.loading_label.config(text="Error: Could not open video file", foreground="red")
    
    def display_image(self, image_path):
        """Display an image on the canvas"""
        try:
            # Load and resize image to fit canvas
            img = Image.open(image_path)
            img = self.resize_to_fit(img)
            
            # Convert to PhotoImage for canvas
            self.tk_img = ImageTk.PhotoImage(img)
            
            # Update canvas
            self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        except Exception as e:
            self.loading_label.config(text=f"Error displaying image: {str(e)}", foreground="red")
    
    def display_frame_image(self, frame):
        """Display a video frame on the canvas"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and resize
            img = Image.fromarray(rgb_frame)
            img = self.resize_to_fit(img)
            
            # Convert to PhotoImage for canvas
            self.tk_img = ImageTk.PhotoImage(img)
            
            # Update canvas
            self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        except Exception as e:
            self.loading_label.config(text=f"Error displaying frame: {str(e)}", foreground="red")
    
    def resize_to_fit(self, img, max_width=600, max_height=500):
        """Resize an image to fit within the specified dimensions while maintaining aspect ratio"""
        original_width, original_height = img.size
        
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height
        
        # Determine new dimensions
        if original_width > max_width or original_height > max_height:
            if aspect_ratio > 1:  # Wider than tall
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:  # Taller than wide
                new_height = max_height
                new_width = int(max_height * aspect_ratio)
            
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        return img
    
    def toggle_play(self):
        """Toggle video playback"""
        if self.video_playing:
            self.video_playing = False
            self.play_btn.config(text="Play")
        else:
            self.video_playing = True
            self.play_btn.config(text="Pause")
            self.play_video()
    
    def play_video(self):
        """Play the video frame by frame"""
        if not self.video_playing or self.video_cap is None:
            return
        
        # Get current position
        current_pos = int(self.video_slider.get())
        
        # Set frame position
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        # Read frame
        ret, frame = self.video_cap.read()
        
        if ret:
            self.current_frame = frame
            self.display_frame_image(frame)
            
            # Update slider
            next_pos = current_pos + 1
            self.video_slider.set(next_pos)
            
            # Schedule next frame
            self.root.after(33, self.play_video)  # ~30fps
        else:
            # End of video
            self.video_playing = False
            self.play_btn.config(text="Play")
    
    def analyze_media(self):
        """Analyze the current media"""
        if not self.system_ready or not self.current_media_path:
            return
        
        # Disable the analyze button during analysis
        self.analyze_btn.config(state=tk.DISABLED)
        self.loading_label.config(text="Analyzing... Please wait.")
        
        # Start analysis in a background thread
        thread = threading.Thread(target=self.perform_analysis)
        thread.daemon = True
        thread.start()
    
    def perform_analysis(self):
        """Perform media analysis in a background thread"""
        try:
            if self.current_media_type == 'image':
                results = self.system.process_image(self.current_media_path)
                
                # Update UI on the main thread
                self.root.after(0, lambda: self.display_image_results(results))
                
            elif self.current_media_type == 'video':
                results = self.system.process_video(self.current_media_path)
                
                # Update UI on the main thread
                self.root.after(0, lambda: self.display_video_results(results))
        
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.root.after(0, lambda: self.loading_label.config(
                text=error_msg, foreground="red"))
            self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL))
    
    def display_image_results(self, results):
        """Display image analysis results"""
        self.current_results = results
        
        # Display annotated image
        annotated_img = Image.fromarray(cv2.cvtColor(results["annotated_image"], cv2.COLOR_BGR2RGB))
        annotated_img = self.resize_to_fit(annotated_img)
        self.tk_img = ImageTk.PhotoImage(annotated_img)
        self.canvas.config(width=self.tk_img.width(), height=self.tk_img.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        
        # Scene description
        self.results_text.insert(tk.END, "SCENE DESCRIPTION:\n", "heading")
        self.results_text.insert(tk.END, f"{results['description']}\n\n")
        
        # Detected objects
        self.results_text.insert(tk.END, "DETECTED OBJECTS:\n", "heading")
        for i, det in enumerate(results["detections"]):
            self.results_text.insert(
                tk.END, 
                f"{i+1}. {det['class_name']} (confidence: {det['confidence']:.2f})\n"
            )
        
        # Configure tags
        self.results_text.tag_configure("heading", font=("Arial", 11, "bold"))
        
        # Enable question answering and save results
        self.ask_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        
        # Update status
        self.loading_label.config(text="Analysis complete!", foreground="black")
        self.analyze_btn.config(state=tk.NORMAL)
    
    def display_video_results(self, results):
        """Display video analysis results"""
        self.current_results = results
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        
        # Video summary
        self.results_text.insert(tk.END, "VIDEO SUMMARY:\n", "heading")
        self.results_text.insert(tk.END, f"{results['summary']}\n\n")
        
        # Keyframe descriptions
        self.results_text.insert(tk.END, "KEYFRAME DESCRIPTIONS:\n", "heading")
        for i, desc in enumerate(results["keyframe_descriptions"]):
            timestamp = results["keyframes"]["timestamps"][i]
            self.results_text.insert(
                tk.END, 
                f"Keyframe {i+1} ({timestamp:.2f}s):\n",
                "subheading"
            )
            self.results_text.insert(tk.END, f"{desc}\n\n")
        
        # Configure tags
        self.results_text.tag_configure("heading", font=("Arial", 11, "bold"))
        self.results_text.tag_configure("subheading", font=("Arial", 10, "bold"))
        
        # Load keyframes for browsing
        self.keyframes = results["keyframes"]["frames"]
        self.keyframe_index = 0
        
        if self.keyframes and len(self.keyframes) > 0:
            self.display_frame_image(self.keyframes[0])
            
            # Enable question answering and save results
            self.ask_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
        
        # Update status
        self.loading_label.config(text="Analysis complete!", foreground="black")
        self.analyze_btn.config(state=tk.NORMAL)
    
    def ask_question(self):
        """Ask a question about the current media"""
        if not self.current_media_path or not self.system_ready:
            return
        
        question = self.question_entry.get().strip()
        if not question:
            return
        
        # Display the question
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, f"Q: {question}\n\n", "question")
        self.answer_text.insert(tk.END, "Thinking...\n", "thinking")
        
        # Disable the ask button while processing
        self.ask_btn.config(state=tk.DISABLED)
        
        # Process in background thread
        thread = threading.Thread(target=self.process_question, args=(question,))
        thread.daemon = True
        thread.start()
    
    def process_question(self, question):
        """Process a question in a background thread"""
        try:
            if self.current_media_type == 'image':
                # Use the current image
                image_source = self.current_media_path
            else:
                # Use the current frame from video
                image_source = self.current_frame
            
            # Get answer using vision-language pipeline
            answer, confidence, all_answers = self.system.vlm.answer_question(
                image_source, question
            )
            
            # Update UI on the main thread
            self.root.after(0, lambda: self.display_answer(answer, confidence, all_answers))
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            self.root.after(0, lambda: self.answer_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.answer_text.insert(tk.END, error_msg, "error"))
            self.root.after(0, lambda: self.ask_btn.config(state=tk.NORMAL))
    
    def display_answer(self, answer, confidence, all_answers):
        """Display the answer to a question"""
        self.answer_text.delete(1.0, tk.END)
        
        # Get the question back
        question = self.question_entry.get().strip()
        
        # Display Q&A
        self.answer_text.insert(tk.END, f"Q: {question}\n\n", "question")
        self.answer_text.insert(tk.END, f"A: {answer} ", "answer")
        self.answer_text.insert(tk.END, f"(confidence: {confidence:.2f})\n\n", "confidence")
        
        # Display other possible answers
        self.answer_text.insert(tk.END, "Other possible answers:\n", "subheading")
        for ans, conf in all_answers.items():
            if ans != answer:  # Skip the top answer
                self.answer_text.insert(tk.END, f"- {ans}: {conf:.2f}\n")
        
        # Configure tags
        self.answer_text.tag_configure("question", font=("Arial", 10, "bold"))
        self.answer_text.tag_configure("answer", font=("Arial", 10))
        self.answer_text.tag_configure("confidence", font=("Arial", 9, "italic"))
        self.answer_text.tag_configure("subheading", font=("Arial", 9, "bold"))
        self.answer_text.tag_configure("error", foreground="red")
        self.answer_text.tag_configure("thinking", foreground="blue")
        
        # Enable the ask button again
        self.ask_btn.config(state=tk.NORMAL)

def main():
    root = tk.Tk()
    app = RobotVisionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()