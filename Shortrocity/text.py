import math
import cv2
import json
import os
import subprocess
from lib.audio_utils import get_audio_duration
from pydub import AudioSegment

"""
This script reads a video file, processes it frame by frame, and writes the frames to a new video file.
It uses OpenCV (cv2) for video processing.
"""


def write_text(text, frame, video_writer_out):
    """
    Write text to a frame and write the frame to a video writer.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    thickness = 10
    fontScale = 3
    border = 10
    border_color = (0, 0, 0)
    
    # calculate the position of the centered text
    text_size = cv2.getTextSize(text, font, fontScale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2 # center horizontally
    text_y = (frame.shape[0] + text_size[1]) // 2 # center vertically
    org = (text_x, text_y)
    
    frame = cv2.putText(frame, text, org, font, fontScale, border_color, thickness+border, cv2.LINE_AA)
    frame = cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    video_writer_out.write(frame)

def get_text_narrations()->str:
    """
    Get the text narrations from the data.json file.
    """
    with open('data.json') as f:
        narration_data = json.load(f) 
    narrations = []
    for element in narration_data:
        if element['type'] == 'text':
            narrations.append(element['content'])
    return narrations

def add_narration_to_video(input_video, output_video_path, frame_rate:int=30):
    """
    Add audio narration and text captions to a video.
    """    
    text_offset = 50 # ms to offset the text from the start of the narration
    
    # Open the input video file
    cap = cv2.VideoCapture(input_video)

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    temp_video_path = "capted_output_video.avi"
    out_video_writer = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    full_narration = AudioSegment.empty()
    narrations = get_text_narrations()
    for i, narration in enumerate(narrations):
        narration_path = os.path.join('narrations',f"narration_{i+1}.mp3")
        duration = get_audio_duration(narration_path)
        narration_frames_duration = int(duration/1000*frame_rate)
        
        print(f'\n\n {narration_path=} {duration=} {narration_frames_duration=} {narration=}');
        print('\n ============\n\n');
        full_narration += AudioSegment.from_file(narration_path)

        char_count = len(narration.replace(" ",''))
        ms_per_char = duration/char_count
        
        frames_written = 0
        words = narration.split(' ')
        for w, word in enumerate(words):
            word_ms = len(word)*ms_per_char
            if i == 0 and w == 0:
                word_ms = max(0, word_ms - text_offset)
                
            for _ in range(math.floor(int(word_ms)/1000*frame_rate)):
                ret, frame = cap.read() 
                if not ret:
                    break
                # Add text caption to the frame
                write_text(word,frame,out_video_writer)
                frames_written += 1
        for _ in range(narration_frames_duration-frames_written):
            ret, frame = cap.read() 
            if not ret:
                break
            out_video_writer.write(frame)

    # add the remaining frames to the video
    while out_video_writer.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out_video_writer.write(frame)
    
    temp_narration_audio_file_path = "narration.mp3"
    full_narration.export(temp_narration_audio_file_path, format='mp3')
    
    # Release the video capture and writer objects (this part should be  outside the while loop)
    cap.release()
    out_video_writer.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

    ffmpeg_command = [
        'ffmpeg',
        '-y',
        '-i', temp_video_path,
        '-i', temp_narration_audio_file_path,
        '-map','0:v', # map video from first input
        '-map','1:a', # map audio from second input
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict','experimental',
        output_video_path
    ]
    subprocess.run(ffmpeg_command)
    
    os.remove(temp_video_path)    
    os.remove(temp_narration_audio_file_path)


# add_narration_to_video('hwfinal_video.avi','vertical_video.avi')