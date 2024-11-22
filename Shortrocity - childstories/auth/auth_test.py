from api_manager import APIKeyManager

api_manager = APIKeyManager()

api_key = api_manager.get_working_elevenlabs_key()
print(f'\n\n {api_key=}');
print('\n ============\n\n');
