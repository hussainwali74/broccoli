from pydub import AudioSegment

def get_audio_duration(audio_path):
    """
    Get the duration of an audio file in milliseconds.

    This function uses the pydub library to load an audio file and return its duration.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        int: The duration of the audio file in milliseconds.

    Note:
        This function supports various audio formats that pydub can handle,
        including mp3, wav, ogg, etc.
    """
    return len(AudioSegment.from_file(audio_path))
