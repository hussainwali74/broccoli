import os
import json
import image_gen
import narration
import video
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# ---------------------------------
client = OpenAI()
with open("source_material.txt", "r") as file:
    source_material = file.read()
    logging.info("Source material loaded.")

messages = [
    {
        "role": "system",
        "content": """You are a Youtube short narrator. You generate 30 seconds to 1 minute long narrations. 
         The shorts you create have a background image that fades from image to image as the narration is going on. 
         Each narration text should be maximum 2 short sentences.
         Respond in the following format, repeated until the end of the narration:
         [Description of the background image]
         Narrator: [Your narration]
         
         You should add a description of the background image that is relevant to the narration. it will be used to generate images later.
         """,
    },
    {
        "role": "user",
        "content": f"create a youtube short narration based on the following source material \n\n {source_material}",
    },
]

response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
response = response.choices[0].message.content
logging.info("Received response from OpenAI.")

with open("narration.txt", "w") as file:
    file.write(response)
    logging.info("Narration written to narration.txt.")
# ---------------------------------


with open('narration.txt', 'r') as file:
    response = file.read()
data = narration.parse(response)
logging.info(f"Parsed narration data: {data}")
with open("data.json", "w") as file:
    json.dump(data, file)
    logging.info("Data written to data.json.")

# with open('data.json', 'r') as file:
#     data = json.load(file)

narration.create(data, "narrations")
# logging.info("Narration audio created.")

 
# generate images
image_gen.create_from_data(data)
logging.info("Images generated from data.")

# generate video, with captions
video.create("zeus.avi")
logging.info("Final video created: hwfinal_video.avi")
