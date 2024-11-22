from PyYoutube.upload_video import upload_video
from prepare_vid import prepare_videos



channels = ['GennyWorld', 'EatonWorld', 'sportsplanetx', 'FableFairyland']

channel_name = 'FableFairyland'

if __name__ == "__main__":
    prepare_videos(channel_name)
    upload_video(channel_name)
    