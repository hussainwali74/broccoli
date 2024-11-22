# vid.py

from moviepy.editor import VideoFileClip
import os

class VideoProcessor:
    def __init__(self, video_file_path):
        """Initialize the video processor."""
        if not os.path.isfile(video_file_path):
            raise FileNotFoundError(f"The file {video_file_path} does not exist.")
        
        # Load the video
        self.clip = VideoFileClip(video_file_path)
        
        # Store original properties
        self.original_fps = self.clip.fps
        self.original_duration = self.clip.duration
        self.original_size = self.clip.size
        
        # Print properties for verification
        print("Video Properties:")
        print(f"- Path: {video_file_path}")
        print(f"- FPS: {self.original_fps}")
        print(f"- Duration: {self.original_duration} seconds")
        print(f"- Size: {self.original_size}")

    def save_video(self, output_path):
        """Save the video to a file."""
        try:
            # Use the original video's properties
            self.clip.write_videofile(
                output_path,
                fps=self.original_fps,  # Use original FPS
                codec='libx264',
                audio_codec='aac',
                remove_temp=True,
                verbose=False
            )
            print(f"Video saved successfully to: {output_path}")
            
        except Exception as e:
            print(f"Error saving video: {str(e)}")
            raise
        finally:
            # Clean up
            try:
                self.clip.close()
            except:
                pass

def test_video_processing():
    """Test function to verify video processing"""
    input_path = 'a.mp4'
    output_path = 'a_processed.mp4'
    
    try:
        # Process video
        processor = VideoProcessor(input_path)
        processor.save_video(output_path)
        
        # Verify output
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path)
            print(f"Output file size: {output_size / (1024*1024):.2f} MB")
            return True
        return False
    
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_video_processing()






























































