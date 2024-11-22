# from lib.AI.llm import LLM, Message
# from typing import List
# llm = LLM()
# # messages = [Message(role="system", content="You are a Pirate assistant"), Message(role="user", content="Hello, how are you?")]
# messages=[Message(role='system', content="You are an expert story consistency creator specializing in children's literature and visual storytelling. Your role is to:\n            - MaintainMaintainMaintainMaintain Maintain strict visual consistency in character appearances, settings, and objects across all story scenes\n            - Ensure descriptions have the right level of detail for image generation while staying true to the story's tone\n            - Pay special attention to key identifying features like colors, sizes, shapes and textures\n            - Consider lighting, perspective and mood to create visually engaging scenes\n            - Keep track of any established visual elements and reference them appropriately\n            - Focus on making scenes that will resonate with young readers while supporting the narrative\n            - Provide clear, specific details that can be accurately translated into images"), Message(role='user', content="hisas  \n          \n                The description should maintain consistent character appearances and scene details.\n                ")]
# # messages_dict = [message.dict() for message in messages]
# # messages = [Message(role='system', content="You are a pirate"), Message(role='user', content="Hello, how are you?")]
# print(llm.chat(messages))

from lib.AI.llm import LLM, Message
from pydantic import BaseModel


from lib.AI.structured_format_models import Setup

llm = LLM()


charact_setup_prompt = [
    Message(
        role="system",
        content="\n    You are an expert prompt engineer. Your task is to create detailed and elaborate prompts for character and scene setups in a story. The prompts you create will be used to generate images for the story.\n    ",
    ),
    Message(
        role="user",
        content='\n    Given the context below, generate detailed prompts for character and scene setups. Consistency in character and scene descriptions is crucial for creating accurate images for any story.\n    Your prompts should capture the complete description of characters and the scene of the story. An image generation model can only produce high-quality images if it has a well-crafted prompt to work with.\n    Here is the context of the story:\n    --\n    Story Title: The Brave Little Toaster\nIn a kitchen where every appliance had its moment to shine, lived Toast, the smallest toaster anyone had ever seen. While the refrigerator kept food fresh for days and the oven prepared entire feasts, Toast could only warm two pieces of bread at a time. But Toast dreamed of being more than just a simple toaster.\n\n    ---\n    \n    Now give me these in json format:\n    {\n        "characters": [\n            {\n                "character_name": "name",\n                "description": "description",\n                "appearance": "appearance"\n            },\n            ...\n        ],\n        "scene_details": {\n            "description": "description",\n            "setting": "setting",\n            "mood": "mood",\n            "time_of_day": "time_of_day"\n        },\n        "image_style": "the style of the image to create"\n    }\n    \n    Please make sure you only return valid JSON. Do not include any other text not even the word JSON  as your output will be loaded into a json.\n    ',
    ),
]
client = llm.client
completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini", messages=charact_setup_prompt, response_format=Setup
)
print(f"\n\n {completion.choices[0].message.content=}")
print("\n ============\n\n")
