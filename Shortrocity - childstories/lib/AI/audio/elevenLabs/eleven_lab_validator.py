from elevenlabs.client import ElevenLabs
from auth.models import APIKeyStatus
class ElevenLabsValidator:
    @staticmethod
    def validate_key(api_key:str) -> bool:
        """Validate an ElevenLabs API key."""
        try:
            eleven_labs = ElevenLabs(api_key=api_key)
            eleven_labs.get_voices()
            return True
        except Exception as e:
            return False
    
    @staticmethod
    def get_key_status(api_key:str) -> str:
        """Get the status of an ElevenLabs API key."""
        try:
            client = ElevenLabs(api_key=api_key)
            client.voices.get_all()
            return APIKeyStatus.VALID
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg:
                return APIKeyStatus.RATE_LIMITED
            elif "unauthorized" in error_msg:
                return APIKeyStatus.INVALID
            return APIKeyStatus.UNKNOWN