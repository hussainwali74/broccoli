import os
import json
import cohere
import logging
import shutil
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from lib.utils import safe_title


def prepare_videos(channel_name):
    """
    You should add videos directly into the videos folder, and then run this script to prepare them for uploading.
    This script will create a folder for each video, and create a video_data.json file with the video metadata.
    Add video details to the video_data.json file.
    """
    try:
        logger.info(f"Starting video preparation for channel: {channel_name}")
        videos_dir = os.path.join(os.getcwd(), "channels", channel_name, "videos")
        if not os.path.exists(videos_dir):
            logger.error(f"Videos directory not found at path: {videos_dir}")
            raise Exception(
                f"Videos directory does not exist for channel: {channel_name}"
            )

        # Get list of files (not directories) in videos_dir
        # Using list comprehension to filter out directories and only keep files
        # This ensures we only process actual video files and not subdirectories
        videos = [
            f
            for f in os.listdir(videos_dir)  # List all items in directory
            if os.path.isfile(os.path.join(videos_dir, f))  # Check if item is a file
        ]

        for video_file in videos:
            try:
                logger.info(f"Processing video file: {video_file}")
                video_title = os.path.splitext(video_file)[0]
                safe_video_title = safe_title(video_title)
                
                # Check if video was already processed or if video_data.json exists and video already uploaded
                video_data_json_file = [f for f in os.listdir(os.path.join(videos_dir, safe_video_title)) if f.endswith('.json')][0]
                video_data_path = os.path.join(videos_dir, safe_video_title, video_data_json_file)
                if os.path.exists(video_data_path):
                    logger.info(f"Video {video_file} was already processed, skipping...\n")
                    continue

                video_folder = os.path.join(videos_dir, safe_video_title)

                if not os.path.exists(video_folder):
                    logger.info(f"Creating video folder: {video_folder}")
                    os.makedirs(video_folder)

                video_data = {
                    "title": video_title,
                    "description": "",
                    "video_file_name": video_file,
                    "uploaded": False,
                    "tags": [],
                }

                # Save video metadata
                video_data_path = os.path.join(video_folder, "video_data.json")
                logger.info(f"Saving video metadata to: {video_data_path}")
                with open(video_data_path, "w") as f:
                    json.dump(video_data, f)

                # Move video file
                video_dest_path = os.path.join(video_folder, video_file)
                logger.info(f"Moving video file to: {video_dest_path}")
                shutil.move(
                    os.path.join(videos_dir, video_file),
                    video_dest_path
                )
                
                # Create thumbnail
                # thumbnail_path = os.path.join(video_folder, "thumbnail.jpg")
                # if not os.path.exists(thumbnail_path):
                #     create_thumbnail(video_dest_path, thumbnail_path)

                logger.info(f"Successfully prepared video: {video_file}")

            except Exception as e:
                logger.error(f"Failed to process video {video_file}: {str(e)}", exc_info=True)
        
        logger.info(f"Starting video details generation for channel: {channel_name}")
        logger.info("This process may take several minutes depending on the number of videos")
        add_videos_details(channel_name)
        
        logger.info(f"Completed video preparation for channel: {channel_name}")
        
    except Exception as e:
        logger.error(f"Critical error preparing videos for channel {channel_name}: {str(e)}", exc_info=True)


def add_videos_details(channel_name, context=None):
    """
    For all the videos in the channel folder, fill the description and SEO tags using Cohere LLM, and update the title.
    The LLM should be used to generate the description and the hashtags. It should act as a cool sports YouTuber and must include emojis to make it cool.
    If provided, use the context to generate the description and the hashtags, otherwise use the video title.
    Update the video_data.json file with the new title, description, and hashtags.

    Args:
        channel_name (str): The name of the YouTube channel.
        context (str, optional): Additional context to generate the description and hashtags. Defaults to None.
    """
    logger.info(f"Starting to add video details for channel: {channel_name}")
    
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        logger.error("COHERE_API_KEY is not set")
        raise ValueError("COHERE_API_KEY is not set")

    co = cohere.ClientV2(api_key=api_key)
    logger.info("Successfully initialized Cohere client")

    videos_dir = os.path.join(os.getcwd(), "channels", channel_name, "videos")
    if not os.path.exists(videos_dir):
        logger.error(f"Videos directory does not exist for channel: {channel_name}")
        raise Exception(f"Videos directory does not exist for channel: {channel_name}")

    for root, dirs, files in os.walk(videos_dir):
        for dir_name in dirs:
            video_data_path = os.path.join(videos_dir, dir_name, "video_data.json")
            if os.path.exists(video_data_path):
                logger.info(f"Processing video in directory: {dir_name}")
                
                with open(video_data_path, "r") as f:
                    video_data = json.load(f)
                if "uploaded" in video_data and video_data["uploaded"] is True:
                    logger.info(f"Video {dir_name} was already uploaded, skipping...")
                    continue
                
                title = video_data["title"]
                prompt = context if context else title
                logger.info(f"Generating content for video: {title}")

                messages = [
                    {
                        "role": "system",
                        "content": f"Act as a hip and most cool youtuber.",
                    },
                    {
                        "role": "user",
                        "content": f"Generate a JSON object with the following fields: 'title', 'description', and 'tags'. This is for a YouTube video titled '{prompt}' in the sports category. The description should be engaging and include emojis. Use Gen-Z slang and language, but do not overdo it. The original title is most of the times mocking or irony to be funny for the reader, so do not take it literally, Try to get context from it. The original title is provides context about the video, so use it for inspiration. The description should be at least 100 words long. Do not use the following words: 'Masterclass', 'explore', 'Unlock'.",
                    },
                ]

                try:
                    response = co.chat(
                        model="command-r-plus-08-2024",
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                    logger.info("Successfully received response from Cohere")

                    # Add debug logging
                    generated_text = response.message.content[0].text.strip()
                    logger.debug(f"Raw response: {generated_text}")
                    
                    try:
                        # Clean up any potential escape characters
                        generated_text = generated_text.encode().decode('unicode-escape')
                        generated_json = json.loads(generated_text)
                        logger.info("Successfully parsed JSON response")
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                        logger.error(f"Problematic text: {generated_text}")
                        continue  # Skip this video and continue with the next one
                    
                    title, description, hashtags = generated_json["title"], generated_json["description"], generated_json["tags"]

                    video_data["title"] = title
                    video_data["description"] = description
                    video_data["tags"] = hashtags

                    with open(video_data_path, "w") as f:
                        json.dump(video_data, f, indent=4)

                    logger.info(f"Successfully updated video details for: {title}")

                except Exception as e:
                    logger.error(f"Error processing video {title}: {e}")

logger.info("Completed adding video details for all videos")

# if __name__ == "__main__":
#     add_videos_details("sportsplanetx")
