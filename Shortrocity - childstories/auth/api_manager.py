import json
import os
from typing import Optional, Dict, List
from datetime import datetime
from lib.AI.audio.elevenLabs.eleven_lab_validator import ElevenLabsValidator
from auth.models import APIKeyStatus
class APIKeyManager:
    def __init__(self, keys_file: str = "auth/keys.json"):
        self.keys_file = os.path.join(os.getcwd(), keys_file)
        self.keys_data = self._load_keys()
        self.current_indices = {service: 0 for service in self.keys_data.keys()}
        
    def _load_keys(self) -> Dict:
        """Load API keys from the JSON file."""
        if not os.path.exists(self.keys_file):
            raise FileNotFoundError(f"Keys file not found: {self.keys_file}")
        
        with open(self.keys_file, 'r') as f:
            return json.load(f)
    
    def _save_keys(self) -> None:
        """Save the current state of API keys back to the JSON file."""
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys_data, f, indent=4)

    def get_next_key(self, service: str, category: str) -> Optional[str]:
        """Get the next available API key for the specified service."""
        if service not in self.keys_data:
            raise KeyError(f"Service '{service}' not found in keys data")
            
        keys_list = self.keys_data[service][category]
        if not keys_list:
            return None

        # Try to find the next unused key
        start_index = self.current_indices.get(service, 0)
        current_index = start_index
        
        while True:
            key_data = keys_list[current_index]
            if not key_data.get("used", False):
                self.current_indices[service] = (current_index + 1) % len(keys_list)
                return key_data["api_key"]
            
            current_index = (current_index + 1) % len(keys_list)
            if current_index == start_index:
                # We've checked all keys and they're all used
                # Reset all keys to unused and start over
                raise Exception(f"No available keys for service {service} and category {category}")

    def _reset_used_status(self, service: str, category: str) -> None:
        """Reset the used status of all keys for a service."""
        for key_data in self.keys_data[service][category]:
            key_data["used"] = False
        self._save_keys()

    def mark_key_status(self, service: str, category: str, api_key: str, status: str) -> None:
        """Mark an API key as used, invalid, or rate limited."""
        for key_data in self.keys_data[service][category]:
            if key_data["api_key"] == api_key:
                key_data["used"] = True
                key_data["last_status"] = status
                key_data["last_checked"] = datetime.now().isoformat()
                break
        self._save_keys()
    
    # Todo: when credits finished only then switch to next key
    def validate_elevenlabs_key(self, api_key: str) -> str:
        """Validate an ElevenLabs API key."""
        return ElevenLabsValidator.get_key_status(api_key)

    def get_working_elevenlabs_key(self) -> Optional[str]:
        """Get a working ElevenLabs API key."""
        api_key = self.get_next_key("audio", "elevenlabs")
        if not api_key:
            return None
        status = APIKeyStatus.VALID
        # Todo: when credits finished only then switch to next key
        # status = self.validate_elevenlabs_key(api_key)
        # self.mark_key_status("audio", "elevenlabs", api_key, status)
        
        if status == APIKeyStatus.VALID:
            return api_key
        return self.get_working_elevenlabs_key()  # Recurs