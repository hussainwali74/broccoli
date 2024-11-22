# basic color filters
import cv2
import numpy as np

def apply_filter(frame, filter_type):
    if filter_type == "grayscale":
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif filter_type == "sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        return cv2.transform(frame, kernel)
    elif filter_type == "warm":
        frame = frame.astype(float)
        frame[:,:,0] *= 0.75  # Reduce blue
        frame[:,:,2] *= 1.25  # Increase red
        return np.clip(frame, 0, 255).astype(np.uint8)
    return frame

# ----
# Instagram-like Filters:
def instagram_filters(frame, filter_type):
    if filter_type == "nashville":
        # Add a warm temperature
        frame = cv2.addWeighted(frame, 0.9, 
                               np.ones_like(frame) * np.array([255, 247, 0]), 0.1, 0)
        return frame
    
    elif filter_type == "toaster":
        # Add contrast and warmth
        frame = cv2.addWeighted(frame, 1.5, 
                               np.zeros_like(frame), 0, 0)
        frame = cv2.addWeighted(frame, 0.9, 
                               np.ones_like(frame) * np.array([0, 0, 255]), 0.1, 0)
        return frame
# ----
# Advanced Effects:
def advanced_effects(frame, effect_type):
    if effect_type == "blur":
        return cv2.GaussianBlur(frame, (15, 15), 0)
    
    elif effect_type == "sharpen":
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(frame, -1, kernel)
    
    elif effect_type == "edge_detection":
        return cv2.Canny(frame, 100, 200)
    
    elif effect_type == "vignette":
        rows, cols = frame.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, 200)
        kernel_y = cv2.getGaussianKernel(rows, 200)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.linalg.norm(kernel)
        output = np.copy(frame)
        for i in range(3):
            output[:,:,i] = output[:,:,i] * mask
        return output
# Apply to Video:
def process_video(input_path, output_path, filter_type):
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply filter
        filtered_frame = apply_filter(frame, filter_type)
        
        # Write the frame
        out.write(filtered_frame)
        
    cap.release()
    out.release()

# Usage
process_video('input.mp4', 'output.mp4', 'sepia')
# --------
# real time preview
def preview_filters():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam
    filters = ["normal", "grayscale", "sepia", "warm", "blur", "sharpen"]
    current_filter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Apply current filter
        filtered = apply_filter(frame, filters[current_filter])
        
        # Show filter name
        cv2.putText(filtered, f"Filter: {filters[current_filter]}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Preview', filtered)
        
        # Press 'n' to switch filters, 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            current_filter = (current_filter + 1) % len(filters)
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# ---
# artistic filters
import torch
import torchvision.transforms as transforms
from PIL import Image

def artistic_filters(frame, style):
    if style == "cartoon":
        # Reduce colors
        n_colors = 8
        data = np.float32(frame).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        cartoon = res.reshape((frame.shape))
        
        # Add edges
        edges = cv2.adaptiveThreshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                                    255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 9, 2)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return cv2.bitwise_and(cartoon, edges)
# to use them in your project
# Example usage
input_video = "input.mp4"
output_video = "output.mp4"

# Apply single filter
process_video(input_video, output_video, "sepia")

# Preview filters
preview_filters()

# Apply multiple filters to same video
filters = ["grayscale", "sepia", "warm", "blur"]
for filter_type in filters:
    output = f"output_{filter_type}.mp4"
    process_video(input_video, output, filter_type)
    

# ---
import cv2
import numpy as np
from typing import Tuple

class VideoEffects:
    @staticmethod
    def zoom(frame: np.ndarray, factor: float, center: Tuple[int, int] = None) -> np.ndarray:
        """
        Zoom in/out effect
        factor > 1 zooms in, factor < 1 zooms out
        """
        if center is None:
            center = (frame.shape[1] // 2, frame.shape[0] // 2)
            
        matrix = cv2.getRotationMatrix2D(center, 0, factor)
        result = cv2.warpAffine(
            frame, matrix, (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return result

    @staticmethod
    def pan(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """
        Pan effect (move frame horizontally/vertically)
        """
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        result = cv2.warpAffine(
            frame, matrix, (frame.shape[1], frame.shape[0]),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return result

    @staticmethod
    def rotate(frame: np.ndarray, angle: float, scale: float = 1.0) -> np.ndarray:
        """
        Rotate frame around its center
        """
        center = (frame.shape[1] // 2, frame.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        result = cv2.warpAffine(
            frame, matrix, (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return result

def apply_effects_to_video(input_path: str, output_path: str):
    """
    Apply various effects to a video
    """
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Calculate effect parameters based on frame position
        progress = frame_count / total_frames
        
        # Example: Zoom in and out effect
        zoom_factor = 1 + 0.5 * np.sin(progress * 2 * np.pi)  # Oscillates between 0.5 and 1.5
        frame = VideoEffects.zoom(frame, zoom_factor)
        
        # Example: Rotating effect
        angle = progress * 360  # Full rotation over video duration
        frame = VideoEffects.rotate(frame, angle)
        
        # Example: Panning effect
        dx = int(50 * np.sin(progress * 2 * np.pi))  # Oscillate horizontally
        dy = int(30 * np.cos(progress * 2 * np.pi))  # Oscillate vertically
        frame = VideoEffects.pan(frame, dx, dy)
        
        out.write(frame)
        frame_count += 1
        
        # Print progress
        if frame_count % 30 == 0:  # Update every 30 frames
            print(f"Processing: {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
    
    cap.release()
    out.release()

# More complex effects
class AdvancedVideoEffects:
    @staticmethod
    def ken_burns(frame: np.ndarray, progress: float) -> np.ndarray:
        """
        Ken Burns effect (slow zoom and pan)
        """
        zoom_factor = 1 + (progress * 0.3)  # Zoom in 30% over duration
        dx = int(progress * 100)  # Pan 100 pixels
        dy = int(progress * 50)   # Pan 50 pixels
        
        frame = VideoEffects.zoom(frame, zoom_factor)
        frame = VideoEffects.pan(frame, dx, dy)
        return frame
    
    @staticmethod
    def whip_pan(frame: np.ndarray, progress: float) -> np.ndarray:
        """
        Fast rotation effect
        """
        angle = progress * 720  # Two full rotations
        blur_amount = int(20 * np.sin(progress * np.pi))  # Motion blur
        
        frame = cv2.GaussianBlur(frame, (blur_amount*2+1, blur_amount*2+1), 0)
        frame = VideoEffects.rotate(frame, angle)
        return frame
    
    @staticmethod
    def dolly_zoom(frame: np.ndarray, progress: float) -> np.ndarray:
        """
        Dolly zoom effect (vertigo effect)
        """
        zoom_in = 1 + (progress * 0.5)
        zoom_out = 1 / (1 + (progress * 0.3))
        
        frame = VideoEffects.zoom(frame, zoom_in)
        frame = cv2.resize(frame, None, fx=zoom_out, fy=zoom_out)
        
        # Crop to original size
        h, w = frame.shape[:2]
        start_x = (w - frame.shape[1]) // 2
        start_y = (h - frame.shape[0]) // 2
        frame = frame[start_y:start_y+h, start_x:start_x+w]
        
        return frame

# Usage example
def create_video_with_effects(input_path: str, output_path: str, effect_type: str = 'zoom'):
    """
    Create a video with the specified effect
    """
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        progress = frame_count / total_frames
        
        if effect_type == 'zoom':
            zoom_factor = 1 + (progress * 0.5)  # Zoom in 50% over duration
            frame = VideoEffects.zoom(frame, zoom_factor)
        elif effect_type == 'ken_burns':
            frame = AdvancedVideoEffects.ken_burns(frame, progress)
        elif effect_type == 'whip_pan':
            frame = AdvancedVideoEffects.whip_pan(frame, progress)
        elif effect_type == 'dolly_zoom':
            frame = AdvancedVideoEffects.dolly_zoom(frame, progress)
            
        out.write(frame)
        
        if frame_count % 30 == 0:
            print(f"Processing: {frame_count}/{total_frames} frames ({(frame_count/total_frames)*100:.1f}%)")
    
    cap.release()
    out.release()

# Example usage
input_video = "input.mp4"
output_video = "output.mp4"

# Apply different effects
effects = ['zoom', 'ken_burns', 'whip_pan', 'dolly_zoom']
for effect in effects:
    output = f"output_{effect}.mp4"
    create_video_with_effects(input_video, output, effect)
    
