import logging
import pyyoutube
from pyyoutube.media import Media
import pyyoutube.models as mds
import os
import json
from .access_token import get_access_token
import shutil
import random
import time
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_video(channel_name):
    """
    Upload videos for a specified YouTube channel.

    This function performs the following steps:
    1. Retrieves the access token for the specified channel.
    2. Initializes a YouTube API client.
    3. Iterates through videos in the channel's directory.
    4. Uploads each video with its associated metadata.

    Args:
        channel_name (str): The name of the YouTube channel.

    Raises:
        Exception: If the video data file is not found for a video.
    """
    logger.info(f"Starting video upload process for channel: {channel_name} \n")
    access_token = get_access_token(channel_name)
    print(f'\n\n {access_token=}');
    print('\n ============\n\n');
    cli = pyyoutube.Client(access_token=access_token.access_token)
    
    videos_dir = os.path.join(os.getcwd(), "channels", channel_name, "videos")
    videos = os.listdir(videos_dir)
    
    for video_title_folder in videos:
        
        logger.info(f"Processing video folder: {video_title_folder}")
        video_folder = os.path.join(videos_dir, video_title_folder)
        json_files = [f for f in os.listdir(video_folder) if f.endswith('.json')]
        if len(json_files) == 0:
            logger.error(f"No JSON file found in {video_folder}")
            continue
        video_data_path = os.path.join(video_folder, json_files[0])
        #
        print(f'\n\n {video_data_path=}');
        print('\n ============\n\n');
        if os.path.exists(video_data_path):
            with open(video_data_path, "r") as f:
                video_data = json.load(f)
            logger.info(f"Loaded video data for: {video_title_folder}")
        else:
            error_msg = f"Video data file not found for {video_title_folder}"
            logger.error(error_msg)
            continue
        if "uploaded" in video_data and video_data["uploaded"] is True:
            logger.info(f"Video already uploaded: {video_title_folder}")
            move_uploaded_folder(channel_name, video_title_folder)
            continue
        else:
            body = mds.Video(
                snippet=mds.VideoSnippet(title=video_data["title"], description=video_data["description"])
            )
            
            video_file_path = os.path.join(video_folder, video_data["video_file_name"])
            media = Media(filename=video_file_path)
            logger.info(f"Preparing to upload video: {video_data['video_file_name']}")

            upload = cli.videos.insert(
                body=body, media=media, parts=["snippet"], notify_subscribers=True
            )

            response = None
            logger.info(f"Uploading video {video_title_folder}/{video_data['video_file_name']}...")
            while response is None:
                status, response = upload.next_chunk()
                if status is not None:
                    logger.info(f"Upload progress: {status.progress():.2f}%")

            video = mds.Video.from_dict(response)
            logger.info(f"Video successfully uploaded. Video ID: {video.id}")
            
            #  mark the video as uploaded in the video_data.json file
            video_data["uploaded"] = True
            with open(video_data_path, "w") as f:
                json.dump(video_data, f)
            
            # After successful video upload, set thumbnail if it exists
            thumbnail_path = os.path.join(video_folder, "thumbnail.jpg")  # or .png
            if os.path.exists(thumbnail_path):
                try:
                    logger.info(f"Setting thumbnail for video {video.id}")
                    with open(thumbnail_path, 'rb') as thumbnail_file:
                        cli.thumbnails.set(
                            video_id=video.id,
                            media=Media(filename=thumbnail_path)
                        )
                    logger.info("Thumbnail set successfully")
                except Exception as e:
                    logger.error(f"Failed to set thumbnail: {str(e)}")
        # add random sleep between 1 and 4 minutes
        minutes = random.randint(1, 4)
        logger.info(f"Sleeping for {minutes} minutes before uploading the next video")
        time.sleep(minutes * 60)

def move_uploaded_folder(channel_name, video_title_folder):
    """
    Moves the uploaded video folder from the 'videos' directory to the 'uploaded_videos' directory.

    Args:
        channel_name (str): The name of the channel.
        video_title_folder (str): The title of the video folder to be moved.

    Returns:
        None
    """
    source_folder = os.path.join(os.getcwd(), "channels", channel_name, "videos", video_title_folder)
    destination_folder = os.path.join(os.getcwd(), "channels", channel_name, "uploaded_videos", video_title_folder)
    
    logger.info(f"Moving folder from {source_folder} to {destination_folder}")
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        logger.info(f"Created destination folder: {destination_folder}")
    
    for filename in os.listdir(source_folder):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        shutil.move(source_file, destination_file)
        logger.info(f"Moved file {source_file} to {destination_file}")
    os.rmdir(source_folder)
    logger.info(f"Successfully moved folder {video_title_folder} to uploaded_videos \n\n==========\n\n")

