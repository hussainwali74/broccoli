import os
import uuid
from elevenlabs import VoiceSettings, Client
from auth.api_manager import APIKeyManager

def text_to_speech_file(text: str) -> str:
    # Initialize the API key manager
    api_manager = APIKeyManager()
    
    # Get a working API key
    api_key = api_manager.get_working_elevenlabs_key()
    if not api_key:
        raise Exception("No valid ElevenLabs API keys available")

    # Initialize the ElevenLabs client with the working key
    client = Client(api_key=api_key)

    try:
        # Calling the text_to_speech conversion API with detailed parameters
        response = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2_5",  # use the turbo model for low latency
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        # Generating a unique file name for the output MP3 file
        save_file_path = f"{uuid.uuid4()}.mp3"

        # Writing the audio to a file
        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

        print(f"{save_file_path}: A new audio file was saved successfully!")
        return save_file_path

    except Exception as e:
        # Mark the key as invalid or rate limited based on the error
        api_manager.mark_key_status("audio", "elevenlabs", api_key, "invalid")
        raise e 