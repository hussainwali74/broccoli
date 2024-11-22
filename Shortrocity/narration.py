import os
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    SpeakOptions,
)

load_dotenv()

narration_api = 'deepgram'
def parse(narration):
    """
    Parse the narration text into a structured format.

    Args:
        narration (str): The narration text to be parsed.

    Returns:
        list: A list of dictionaries containing parsed narration data. Each dictionary
              represents either a text element or an image description.
              Example:
              [
                  {"type": "text", "content": "Sample narration text."},
                  {"type": "image", "description": "Description of the background image."}
              ]
    """
    lines = narration.split("\n")
    
    output = []
    for line in lines:
        if line.startswith('Narrator: '):
            text = line.replace('Narrator: ', "")
            output.append({"type": "text", "content": text})
        elif line.startswith("["):
            background_image = line.strip("[]")
            output.append({"type": "image", "description": background_image})
    
    return output


def create(data, output_folder):
    """
    Create audio narrations from text data and save them as MP3 files.

    Args:
        data (list): A list of dictionaries containing narration data.
        output_folder (str): The path to the folder where the audio files will be saved.

    Returns:
        str: An empty string (unused in the current implementation).

    This function processes the input data, generates audio narrations for text elements,
    and saves them as MP3 files in the specified output folder. It uses the Deepgram API
    for text-to-speech conversion.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    narration = ""
    n = 0
    for element in data:
        if element["type"] != "text":
            continue
        n += 1
        output_file = os.path.join(output_folder, f'narration_{n}.mp3')

        try:
            if narration_api == 'deepgram':
                SPEAK_OPTIONS = {"text": element["content"]}
                print('------------------------')
                print(f'\n narration: {narration}')
                print('\n\n\n\n')
                print('------------------------')
                # STEP 1: Create a Deepgram client using the API key from environment variables
                deepgram = DeepgramClient(api_key=os.getenv("DG_API_KEY"))
                
                # STEP 2: Configure the options (such as model choice, audio configuration, etc.)
                options = SpeakOptions(
                    # model="aura-luna-en",
                    # model="aura-orpheus-en",
                    # model="aura-asteria-en",
                    model="aura-angus-en",
                    encoding="linear16",
                    container="wav",
                )

                # STEP 3: Call the save method on the speak property
                response = deepgram.speak.rest.v("1").save(output_file, SPEAK_OPTIONS, options)
                print(response.to_json(indent=4))
            else:
                print('OPENAI not implemented')
        except Exception as e:
            print(f"Exception: {e}")

    return narration
