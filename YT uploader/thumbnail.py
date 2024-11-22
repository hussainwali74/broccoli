import cv2
from PIL import Image
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def create_thumbnail(video_path, output_path, num_samples=10):
    """
    Create a thumbnail by sampling multiple frames from the video and selecting the best one.
    
    Args:
        video_path (str): Path to the video file
        output_path (str): Path where thumbnail should be saved
        num_samples (int): Number of frames to sample from the video
    
    Returns:
        bool: True if thumbnail was created successfully, False otherwise
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    # Calculate frame positions to sample
    sample_positions = [
        int((duration * i / (num_samples - 1)) * fps)
        for i in range(num_samples)
    ]
    
    best_frame = None
    best_score = -float('inf')
    
    for frame_pos in sample_positions:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            continue

        # Calculate frame quality score
        score = calculate_frame_quality(frame)
        
        if score > best_score:
            best_score = score
            best_frame = frame
            logger.info(f"New best frame found at position {frame_pos}/{frame_count} with score {score:.2f}")
    
    if best_frame is None:
        logger.error(f"Could not find any valid frames in video: {video_path}")
        return False
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
    
    # Create PIL Image
    image = Image.fromarray(frame_rgb)
    
    # Resize to YouTube thumbnail dimensions (1280x720)
    image = image.resize((1280, 720), Image.Resampling.LANCZOS)
    
    # Save thumbnail
    image.save(output_path, "JPEG", quality=95)
    
    # Release video capture
    cap.release()
    
    logger.info(f"Created thumbnail: {output_path} with quality score: {best_score:.2f}")
    return True

def calculate_frame_quality(frame):
    """
    Calculate a quality score for a frame using SAM2 and other metrics.
    
    Args:
        frame: OpenCV image frame
    
    Returns:
        float: Quality score (higher is better)
    """
    try:
        # Convert to grayscale for basic calculations
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Basic metrics (keeping some from before)
        brightness = np.mean(gray)
        brightness_score = -abs(brightness - 127.5)
        contrast = np.std(gray)
        
        # Use SAM2 to detect objects and their prominence
        sam2_score = analyze_frame_with_sam2(frame)
        
        # Motion blur detection (higher values mean less blur)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Combine scores with weights
        final_score = (
            brightness_score * 0.1 +    # Basic image quality
            contrast * 0.1 +            # Basic image quality
            laplacian_var * 0.2 +       # Sharpness
            sam2_score * 0.6            # Give highest weight to ML-detected content
        )
        
        return final_score
        
    except Exception as e:
        logger.error(f"Error calculating frame quality: {str(e)}")
        return -float('inf')

def analyze_frame_with_sam2(frame):
    """
    Use SAM2 to analyze frame content and return a quality score.
    
    Args:
        frame: OpenCV image frame
    
    Returns:
        float: Content quality score based on SAM2 analysis
    """
    try:
        # Initialize SAM2 model (do this once and reuse)
        if not hasattr(analyze_frame_with_sam2, 'sam2_model'):
            from sam2 import sam2_image_predictor, SAM2
            
            # Use the model from our models directory
            sam_checkpoint = os.path.join(os.getcwd(), "models", "sam2_hiera_large.pt")
            analyze_frame_with_sam2.sam2_model = sam2_model_registry["hiera_large"](
                checkpoint=sam_checkpoint
            )
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Set image in predictor
        analyze_frame_with_sam2.sam2_model.set_image(frame_rgb)
        
        # Get automatic mask predictions
        masks, scores, _ = analyze_frame_with_sam2.sam2_model.predict()
        
        # Calculate score based on:
        # 1. Number of distinct objects detected
        num_objects = len(masks)
        
        # 2. Size and centrality of main objects
        object_scores = []
        height, width = frame.shape[:2]
        center_y, center_x = height // 2, width // 2
        
        for mask, confidence in zip(masks, scores):
            # Calculate object size relative to frame
            size_score = np.sum(mask) / (height * width)
            
            # Calculate center of mass
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                center_y_obj = np.mean(y_indices)
                center_x_obj = np.mean(x_indices)
                
                # Calculate distance from center (normalized)
                dist_from_center = np.sqrt(
                    ((center_y_obj - center_y) / height) ** 2 +
                    ((center_x_obj - center_x) / width) ** 2
                )
                centrality_score = 1 - dist_from_center
                
                # Include confidence score from SAM
                object_scores.append(size_score * centrality_score * confidence)
        
        # Combine scores
        if object_scores:
            # Favor frames with prominent, centered objects
            content_score = (
                0.4 * min(num_objects, 5) / 5 +  # Number of objects (cap at 5)
                0.6 * max(object_scores)         # Best object score
            )
        else:
            content_score = 0
            
        return content_score * 1000  # Scale up to be comparable with other metrics
        
    except Exception as e:
        logger.error(f"Error in SAM analysis: {str(e)}")
        return 0

video_path = os.path.join(os.getcwd(),'channels', 'sportsplanetx', 'videos','Ancelotti_to_MBappe_AFter_Classico','Ancelotti to MBappe AFter Classico.mp4')
output_path = os.path.join(os.path.dirname(video_path), 'thumbnail.jpg')
create_thumbnail(video_path, output_path)
