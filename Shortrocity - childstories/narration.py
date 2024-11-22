import os
from lib.AI.audio.deepgram import DeepGramGen
from lib.AI.audio.elevenLabs.eleven_lab import HelevenLabs
from lib.utils import wait_for_n_random_minutes
import logging

# narration_api = 'deepgram'

narration_api = 'elevenlabs'
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
    Create audio narrations from text data and save them as MP3 files in output_folder/narrations folder.

    Args:
        data (list): A list of dictionaries containing narration data.
        output_folder (str): The path to the folder where the audio files will be saved.

    Returns:
        str: An empty string (unused in the current implementation).

    This function processes the input data, generates audio narrations for text elements,
    and saves them as MP3 files in the specified output folder. It uses the Deepgram API
    for text-to-speech conversion.
    """
    output_folder = os.path.join(output_folder,'narrations')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    narration = ""
    n = 0
    for i, element in enumerate(data):
        if element["type"] != "text":
            continue
        n += 1
        output_file = os.path.join(output_folder, f'narration_{n}.mp3')
        if not os.path.exists(output_file):
            if narration_api == 'deepgram':
                deepgram = DeepGramGen()
                response = deepgram.text_to_speech(element["content"], output_file)
                print(f'\n\n {response=}');
                print('\n ============\n\n');
                print(response.to_json(indent=4))
            elif narration_api == 'elevenlabs':
                elevenlabclient = HelevenLabs()
                response = elevenlabclient.text_to_speech(element["content"], output_file)
            else:
                print('OPENAI not implemented')
            logging.info(f'Narration {n} of {len(data)} completed.')
            wait_for_n_random_minutes(i)


    return narration
