import cv2 
import numpy as np
import os
import math
import text
import subprocess
from pydub import AudioSegment
from lib.audio_utils import get_audio_duration
from lib.image_utils import resize_image

def write_frames_to_video(image1, image2, out, frame_rate, fade_time, height, width, duration):

    for _ in range(duration):
        vertical_video_frame = np.zeros((height, width, 3), dtype=np.uint8)
        vertical_video_frame[:image1.shape[0], :] = image1
        out.write(vertical_video_frame)
    
    for alpha in np.linspace(0, 1, math.ceil(fade_time/1000*frame_rate)):
        blended_image = cv2.addWeighted(image1, 1-alpha, image2, alpha, 0)
        vertical_video_frame = np.zeros((height, width, 3), dtype=np.uint8)
        vertical_video_frame[:image1.shape[0], :] = blended_image
        out.write(vertical_video_frame)
    return out

def create(output_video_path, narrations_folder, images_folder, text_narration_folder):
    # define the dimensions and frame rate of the video
    width, height = 1080, 1920
    frame_rate = 30

    fade_time = 2000

    # create a video writer object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # YOU can change the codec to mp4v, etc.
    temp_video_path = 'vertical_video.mp4'
    out = cv2.VideoWriter(temp_video_path, fourcc, frame_rate, (width, height))

    # list of image file paths to use in the video
    image_paths = os.listdir(images_folder)
    image_paths.sort()
    # loop through the images and add them to the video
    for i, image in enumerate(image_paths):
        image1_path = os.path.join(images_folder, image_paths[i])
        image1 = cv2.imread(image1_path)
        if i < len(image_paths)-1:
            image2 = cv2.imread(os.path.join(images_folder, image_paths[i+1]))
        else:
            image2 = cv2.imread(os.path.join(images_folder, image_paths[0]))
        
        image1 = resize_image(image1, width, height)
        image2 = resize_image(image2, width, height)
        
        narration = os.path.join(narrations_folder, f'narration_{i+1}.mp3')
        image_display_duration = get_audio_duration(narration)-fade_time
        # if i >0 or i ==len(image_paths)-1:
        #     duration -=fade_time
        
        image_display_duration = math.ceil(image_display_duration/1000*frame_rate)

        write_frames_to_video(image1, image2, out, frame_rate, fade_time, height, width, image_display_duration)


    # release the video writer object
    out.release()
    cv2.destroyAllWindows()
    # text.add_narration_to_video(temp_video_path, output_video_path, narrations_folder, text_narration_folder)
    text.add_narration_only_to_video(temp_video_path, output_video_path, narrations_folder, text_narration_folder)
    os.remove(temp_video_path)
