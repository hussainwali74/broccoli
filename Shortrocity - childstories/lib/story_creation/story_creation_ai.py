import logging
from lib.AI.llm import LLM, Message
from lib.AI.structured_format_models import (
    Setup,
    RevisedImageDescriptions,
    VideoMetadata,
)
import os
import json

logger = logging.getLogger(__name__)


def consistent_scener(story_content):
    """
    Create consistent scene descriptions for story images.

    Args:
        story_content (dict): The full story content with text and image elements

    Returns:
        dict: Story content with consistent image descriptions
    """
    logger.info(
        f"Starting scene consistency generation for story: {story_content['title']}"
    )
    llm = LLM()

    # Extract key story elements for consistency
    title = story_content["title"]

    # Build character and scene descriptions from first few text elements
    context = f"Story Title: {title}\n"
    for element in story_content["elements"][:2]:
        if element["type"] == "text":
            context += element["content"] + "\n"
    charact_setup_prompt = [
        Message(
            role="system",
            content="""
    You are an expert prompt engineer. Your task is to create detailed and elaborate prompts for character and scene setups in a story. The prompts you create will be used to generate images for the story.
    """,
        ),
        Message(
            role="user",
            content=f"""
    Given the context below, generate detailed prompts for character and scene setups. Consistency in character and scene descriptions is crucial for creating accurate images for any story.
    Your prompts should capture the complete description of characters and the scene of the story. An image generation model can only produce high-quality images if it has a well-crafted prompt to work with.
    Here is the context of the story:
    --
    {context}
    ---
    
    Now give me these in json format:
    {{
        "characters": [
            {{
                "character_name": "name",
                "description": "description",
                "appearance": "appearance"
            }},
            ...
        ],
        "scene_details": {{
            "description": "description",
            "setting": "setting",
            "mood": "mood",
            "time_of_day": "time_of_day"
        }},
        "image_style": "the style of the image to create"
    }}
    
    Please make sure you only return valid JSON. Do not include any other text not even the word JSON  as your output will be loaded into a json.
    Make sure the description or appearance is in no way NSFW, or it will be rejected by the image generation model.
    """,
        ),
    ]
    client = llm.client
    response1 = client.beta.chat.completions.parse(
        model="gpt-4o-mini", messages=charact_setup_prompt, response_format=Setup
    )

    setup = response1.choices[0].message.content
    # Update image descriptions to be consistent
    image_count = 0
    messages = [
        Message(
            role="system",
            content="""You are an expert story consistency creator specializing in children's literature and visual storytelling. Your role is to:
- Maintain strict visual consistency in character appearances, settings, and objects across all story scenes
- Ensure descriptions have the right level of detail for image generation while staying true to the story's tone
- Pay special attention to key identifying features like colors, sizes, shapes and textures
- Consider lighting, perspective and mood to create visually engaging scenes
- Keep track of any established visual elements and reference them appropriately
- Focus on making scenes that will resonate with young readers while supporting the narrative
- Provide clear, specific details that can be accurately translated into images
- ALWAYS include the style of image to create.
- There must be no text in the generated image.
i will provide you with an existing image description for each story moment, You have to fix them according the instructions provided here and the image guide provided below.
""",
        ),
    ]

    image_descriptions = ""
    for i, element in enumerate(story_content["elements"]):
        if element["type"] == "image":
            image_descriptions += f"Image {i+1}: {element.get('description', '')}\n"
    message = Message(
        role="user",
        content=f"""
    Based on this story context:
    --
    {context}
    --
    and this image guide:
    --
    {setup}
    --
    Here are the existing image descriptions for each story moment:
    --
    {image_descriptions}
    
    The prompt you generate will be taken as a prompts for an image generation model so be as descriptive as possible, closely align with the image guide.
    Ensure the prompt is optimized for Flux Schnell image generation model.
    Do not forget to explicitly mention the image style provided in the setup, in each prompt so that the image generation model knows what to create.
    """,
    )
    messages.append(message)
    image_descriptions = []
    try:

        client = llm.client
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=RevisedImageDescriptions,
        )
        image_descriptions = response.choices[0].message.parsed.image_descriptions
    except Exception as e:
        logger.error(f"Error generating description for image {image_count}: {str(e)}")
    if image_descriptions:
        for i, element in enumerate(story_content["elements"]):
            if element["type"] == "image":
                element["description"] = image_descriptions[0]
                image_descriptions.pop(0)

    logger.debug(f"Generated description for image {image_count}")

    logger.info(f"Completed scene consistency generation for story: {title}")
    return story_content


def create_video_metadata(
    processed_done_folder,
    story,
    generated_videos_folder,
    out_json_file_name="video.json",
):
    """
    Create video metadata for a story.

    Args:
        processed_done_folder: Path to folder containing processed stories
        story: Name of the story
        generated_videos_folder: Path to folder for generated video metadata
        out_json_file_name: Name of output JSON file (default: video.json)

    Raises:
        FileNotFoundError: If story JSON file or folders cannot be found
        JSONDecodeError: If story JSON file is invalid
        Exception: For other errors during metadata generation
    """
    try:
        story_json_path = os.path.join(processed_done_folder, story)
        if not os.path.exists(story_json_path):
            raise FileNotFoundError(f"Story folder not found: {story_json_path}")

        json_files = [f for f in os.listdir(story_json_path) if f.endswith(".json")]
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {story_json_path}")

        json_file = json_files[0]

        try:
            with open(os.path.join(story_json_path, json_file), "r") as f:
                story_content = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in story file: {json_file}")
            raise

        llm = LLM()
        client = llm.client

        messages = [
            Message(
                role="system",
                content="""You are an expert at creating engaging YouTube video metadata for children's stories. 
Your task is to create a title, description and 20 tags that will help the video reach its target audience of children aged 12 and under.""",
            ),
            Message(
                role="user",
                content=f"""
Given this children's story content, create YouTube video metadata including:
- An engaging, child-friendly title
- A description that summarizes the story and encourages viewing
- 20 relevant tags for discovery, in the tags include the channel name: Fable FairyLand 

Story content:
{json.dumps(story_content, indent=2)}

Return only valid JSON in this format:
{{
"title": "title",
"description": "description", 
"video_file_name": "{story}.mp4",
"uploaded": false,
"tags": ["tag1", "tag2", "tag3"]
}}
""",
            ),
        ]

        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini", messages=messages, response_format=VideoMetadata
            )
            video_metadata = response.choices[0].message.parsed

        except Exception as e:
            logger.error(f"Error generating video metadata: {str(e)}")
            raise

        try:
            # Save video metadata
            video_metadata_dir = os.path.join(generated_videos_folder, story)
            os.makedirs(video_metadata_dir, exist_ok=True)
            video_json_path = os.path.join(video_metadata_dir, out_json_file_name)

            with open(video_json_path, "w") as f:
                json.dump(video_metadata.model_dump(), f, indent=2)

            logger.info(f"Created video metadata file: {video_json_path}")

        except OSError as e:
            logger.error(f"Error saving video metadata file: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error in create_video_metadata: {str(e)}")
        raise
