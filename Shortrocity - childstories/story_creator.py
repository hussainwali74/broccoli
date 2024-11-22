import random
import time
import os
import json
import shutil
import logging
from dotenv import load_dotenv
from Imager.imagen import generate_image
import narration
import video
from lib.story_creation.story_creation_ai import consistent_scener, create_video_metadata

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s\n',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('story_creator.log')
    ],
)

# Get root logger and set level
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# root_logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

def process_stories():
    """Process all stories in the generated folder."""
    logger.info("Starting story processing")

    # Create processed data directory if it doesn't exist
    processed_dir = os.path.join(os.getcwd(), "data", "processed_data", "stories") 
    os.makedirs(processed_dir, exist_ok=True)
    logger.debug(f"Created/verified processed directory: {processed_dir}")
    # Process each story file
    generated_dir = os.path.join(os.getcwd(), "data", "stories_data", "generated")
    d = [f for f in os.listdir(generated_dir)]
    story_files = [f for f in os.listdir(generated_dir) if f.endswith(".json")]
    logger.info(f"Found {len(story_files)} stories to process")

    for filename in story_files:
        logger.info(f"Processing story file: {filename}")
        try:
            # Load story
            original_story_path = os.path.join(generated_dir, filename)
            with open(original_story_path, "r") as f:
                story_content = json.load(f)

            # # Create story directory
            story_name = os.path.splitext(filename)[0]
            processed_story_dir = os.path.join(processed_dir, story_name)
            os.makedirs(processed_story_dir, exist_ok=True)
            logger.debug(f"Created story directory: {processed_story_dir}")

            # Copy original story file
            story_json_path = os.path.join(processed_story_dir, f"{story_name}.json")
            shutil.copy2(original_story_path, story_json_path)
            logger.debug(f"Copied original story file to: {story_json_path}\n")

            # Generate consistent scene descriptions
            logger.debug(f"Generating consistent scene descriptions for story: {story_name}\n")
            updated_content = consistent_scener(story_content)
            
            logger.debug("Generated consistent scene descriptions")

            # Save updated story content
            with open(story_json_path, "w") as f:
                json.dump(updated_content, f, indent=2)
            logger.debug(f"Saved updated story content in: {story_json_path}\n")

            # Generate images
            image_number = 0
            for element in updated_content["elements"]:
                if element["type"] == "image":
                    image_number += 1
                    images_dir = os.path.join(processed_story_dir, "images")
                    os.makedirs(images_dir, exist_ok=True)
                    image_path = os.path.join(images_dir, f"image_{image_number}.png")
                    if not os.path.exists(image_path):
                        logger.info(f'\n\n generating image {image_number} for {story_name}\n\n')
                        generate_image(element["description"], image_path)
            # --------------------------------------------------------------------------------------------------------
            # create narrations
            # --------------------------------------------------------------------------------------------------------
            logger.info(f"creating narrations")
            narration.create(story_content['elements'], processed_story_dir)
                    
            logger.info(f"\n Completed processing story: {story_name}")
            # os.remove(original_story_path)
            break
        except Exception as e:
            logger.error(f"Error processing story {filename}: {str(e)}")
            raise
        break
        logger.info(f"\n------------------\nCompleted processing story: {story_name}\n------------------\n")
        
        # wait for 5 to 10 minutes randomly 
        wait_time = random.randint(300, 600)
        logger.info(f"Waiting for {wait_time} seconds before processing the next story")
        time.sleep(wait_time)
        
if __name__ == "__main__":
    try:
        process_stories()
        # generate video, with captions
        for i in range(1):
            logger.info(f"\n\n -------------------------\n Processing story {i+1} \n------------------------------\n\n")
            processed_stories = os.path.join(os.getcwd(),'data',"processed_data","stories")
            stories = os.listdir(processed_stories)
            story = stories[0]
            
            story_folder_path = os.path.join(processed_stories,story)
            narrations_folder = os.path.join(os.getcwd(),'data',"processed_data","stories",story_folder_path,"narrations")
            images_folder = os.path.join(os.getcwd(),'data',"processed_data","stories",story_folder_path,"images")
            text_narration_folder = os.path.join(os.getcwd(),'data',"processed_data","stories",story_folder_path)
            # video
            generated_video_folder_path = os.path.join(os.getcwd(),'data',"generated_videos", story)
            os.makedirs(generated_video_folder_path, exist_ok=True)
            video_file_name = os.path.join(generated_video_folder_path, story+".mp4")
            video.create(video_file_name, narrations_folder,images_folder, text_narration_folder)
            

            # --------------------------------------------------------------------------------------------------------
            # Clean up
            # --------------------------------------------------------------------------------------------------------
            # move the processed_data/stories/story/ to processed_data/done folder
            logger.info(f"\nMoving story folder to processed_data/done folder: {story_folder_path}\n")
            processed_done_folder = os.path.join(os.getcwd(),'data',"processed_data","done")
            os.makedirs(processed_done_folder, exist_ok=True)
            shutil.move(story_folder_path, processed_done_folder)
            # --------------------------------------------------------------------------------------------------------
            
            
            # --------------------------------------------------------------------------------------------------------
            # Create video metadata for YouTube upload
            # --------------------------------------------------------------------------------------------------------
            logger.info(f"\nCreating video metadata for YouTube upload: {story}\n")
            generated_videos_folder = os.path.join(os.getcwd(),'data',"generated_videos")
            create_video_metadata(processed_done_folder, story, generated_videos_folder)
            # --------------------------------------------------------------------------------------------------------

            # Todo: from the data/generated_videos upload the video to YouTube.
            logger.info(f"\nFinal video created: {story}.mp4\n {generated_video_folder_path=}\n\n-----------------------------------\n")

            # wait for 5 to 10 minutes randomly 
            # wait_time = random.randint(120, 240)
            # logger.info(f"Waiting for {wait_time//60} minutes before processing the next story")
            # time.sleep(wait_time)
            
    except Exception as e:
        logger.error(f"Fatal error in story processing: {str(e)}")
        raise
