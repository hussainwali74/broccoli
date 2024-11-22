from pydantic import BaseModel,Field
from typing import List
class Setup(BaseModel):
    characters_description: str
    scene_details: str
    image_style: str

class RevisedImageDescriptions(BaseModel):
    image_descriptions: List[str] = Field(description="A list of image descriptions, each must include explicitly mentioned image style. The image style must be exactly same in all image_descriptions.")

class VideoMetadata(BaseModel):
    title: str = Field(description="A title for the video, must be short and catchy.")
    description: str = Field(description="A description for the video, must use the video title in the description, must be descriptive of the story and catchy. include hashtags at the end of the description.")
    thumbnail_prompt: str = Field(description="Prompt to generate the thumbnail for the video, This will be used to generate image by Image generation model. I must see the video title in the output image, must be descriptive of the story.")
    video_file_name: str
    uploaded: bool
    tags: List[str] = Field(description="A list of tags, must be more than 10 tags and really relevant to the story, must use the video title in the tags.")