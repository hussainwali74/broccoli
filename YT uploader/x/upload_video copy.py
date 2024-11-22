import os
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import run_flow
from googleapiclient import discovery
import httplib2
from datetime import datetime, timedelta
import pytz
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def getScheduleDateTime(days = 0):
    logger.info(f"Calculating schedule date time for {days} days from now")
    # Set the publish time to 2 PM Eastern Time (US) on the next day
    eastern_tz = pytz.timezone('America/Los_Angeles')
    publish_time = datetime.now(eastern_tz)
    if days>0:
        publish_time = datetime.now(eastern_tz) + timedelta(days)
    publish_time = publish_time.replace(hour=14, minute=0, second=0, microsecond=0)

    # Set the publish time in the UTC timezone
    publish_time_utc = publish_time.astimezone(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    logger.info(f"Scheduled date time: {publish_time_utc}")
    return publish_time_utc

# Start the OAuth flow to retrieve credentials
def authorize_credentials(channel_folder):
    """
    Authorizes credentials for accessing the YouTube API.

    This function performs the following steps:
    1. Fetches credentials from storage credentials.storage in the channel folder.
    2. If credentials are not found or are invalid, it initiates the OAuth flow.
    """
    logger.info(f"Starting authorization process for channel in folder: {channel_folder}")
    CLIENT_SECRET = os.path.join(channel_folder, "client_secret.json")
    SCOPE = "https://www.googleapis.com/auth/youtube"
    # Create a Storage object to manage the storage of OAuth 2.0 credentials
    # The credentials will be stored in a file named "credentials.storage" in the channel folder
    STORAGE = Storage(os.path.join(channel_folder, "credentials.storage"))
    # Fetch credentials from storage
    credentials = STORAGE.get()
    # If the credentials doesn't exist in the storage location then run the flow
    if credentials is None or credentials.invalid:
        logger.info("No valid credentials found. Initiating OAuth flow.")
        flow = flow_from_clientsecrets(CLIENT_SECRET, scope=SCOPE)
        http = httplib2.Http()
        credentials = run_flow(flow, STORAGE, http=http)
    logger.info("Authorization successful")
    return credentials

def getYoutubeService(channel_folder):
    """
    Retrieves an authenticated YouTube API service object.

    This function performs the following steps:
    1. Obtains authorized credentials using the authorize_credentials function.
    2. Authorizes an HTTP object with these credentials.
    3. Builds and returns a YouTube API service object using the authorized HTTP object.

    Returns:
        googleapiclient.discovery.Resource: An authenticated YouTube API service object.

    Raises:
        Any exceptions that may occur during the authorization or service building process.
    """
    logger.info(f"Getting YouTube service for channel in folder: {channel_folder}")
    credentials = authorize_credentials(channel_folder)
    http = credentials.authorize(httplib2.Http())
    discoveryUrl = ('https://www.googleapis.com/discovery/v1/apis/youtube/v3/rest')
    service = discovery.build('youtube', 'v3', http=http, discoveryServiceUrl=discoveryUrl)
    logger.info("YouTube service obtained successfully")
    return service

def upload_video(channel_folder, file_path, title, description='', tags=[], privacy_status='public', day=0):
    logger.info(f"Starting video upload process for '{title}' in channel folder: {channel_folder}")
    youtube = getYoutubeService(channel_folder)
    try:
        # Define the video resource object
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
            },
            'status': {
                'privacyStatus': privacy_status
            }
        }

        if privacy_status == 'private':
            body['status']['publishAt'] = getScheduleDateTime(day)
            logger.info(f"Video set to private with publish date: {body['status']['publishAt']}")

        # Define the media file object
        logger.info(f"Preparing to upload file: {file_path}")
        media_file = MediaFileUpload(file_path)

        # Call the API's videos.insert method to upload the video
        logger.info("Initiating video upload to YouTube")
        videos = youtube.videos()
        response = videos.insert(
            part='snippet,status',
            body=body,
            media_body=media_file
        ).execute()

        # Print the response after the video has been uploaded
        logger.info('Video uploaded successfully!')
        logger.info(f'Title: {response["snippet"]["title"]}')
        logger.info(f'URL: https://www.youtube.com/watch?v={response["id"]}')

    except HttpError as e:
        error_message = f"An HTTP error {e.resp.status} occurred: {e.content.decode('utf-8')}"
        logger.error(error_message)
        raise Exception(error_message)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Usage: python script.py <channel_folder> <video_path>")

    channel_folder = sys.argv[1]
    channel_folder = os.path.join(os.getcwd(),'channels', channel_folder)
    vid_path = sys.argv[2]

    if not os.path.isdir(channel_folder):
        print(f"Error: Channel folder '{channel_folder}' does not exist.")
        sys.exit(1)

    if not os.path.isfile(vid_path):
        print(f"Error: Video file '{vid_path}' does not exist.")
        sys.exit(1)
    logger.info(f"Attempting to upload video: {vid_path} for channel in folder: {channel_folder}")
    upload_video(channel_folder, vid_path, "commando running on air")
    logger.info("Video upload process completed")
