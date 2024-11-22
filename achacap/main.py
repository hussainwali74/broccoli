import whisper
import sys
import json
import torch

import warnings
warnings.filterwarnings("ignore")


# Load the model with weights_only=True
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'\n\n {device=}');
# print('\n ============\n\n');

# model = whisper.load_model("base", download_root=".", device=device)

# audio_file = sys.argv[1]

# result = model.transcribe(
#     audio=audio_file,
#     word_timestamps=True,
#     fp16=False,  # Explicitly set fp16 to False for CPU
# )

# with open('result.json', 'w') as f:
#     json.dump(result, f)



with open('result.json', 'r') as f:
    result = json.load(f)

segments = result['segments']

for segment in segments:
    print(f"{segment['start']} - {segment['end']}: {segment['text']}")
    for word in segment['words']:
        print(f"    {word['start']} - {word['end']}: {word['word']}")
