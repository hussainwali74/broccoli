from deepgram import (
    DeepgramClient,
    SpeakOptions,
)
import os
from dotenv import load_dotenv
from enum import Enum
load_dotenv()


class DeepGramVoices(Enum):
    ANGUS = "aura-angus-en"
    LUNA = "aura-luna-en"
    ORPHEUS = "aura-orpheus-en"
    ASTERIA = "aura-asteria-en"
    ARCAS = "aura-arcas-en"
    ATHENA = "aura-athena-en"


class DeepGramGen:
    def __init__(
        self, api_key: str = None, voice: DeepGramVoices = DeepGramVoices.ARCAS
    ):
        if api_key:
            self.client = DeepgramClient(api_key)
        else:
            self.client = DeepgramClient(
                os.getenv("DG_API_KEY") or os.getenv("DEEPGRAM_API_KEY")
            )
        self.voice = voice

    def text_to_speech(self, text: str, output_file: str) -> str:
        deepgram = self.client

        SPEAK_OPTIONS = {"text": text}

        options = SpeakOptions(
            model=self.voice.value,
            encoding="linear16",
            container="wav",
        )

        try:
            response = deepgram.speak.rest.v("1").save(
                output_file, SPEAK_OPTIONS, options
            )
            return response
        except Exception as e:
            print(f"Error during text-to-speech conversion: {str(e)}")
            return {"error": str(e)}
