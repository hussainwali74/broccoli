import os
import json
import logging
from typing import Union
from pyyoutube import Api
from pyyoutube.models import AccessToken

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Allow insecure transport for development purposes
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

def get_api(channel_name: str) -> Api:
    """
    Create and return an Api instance for the specified channel.

    Args:
        channel_name (str): The name of the channel to create the Api for.

    Returns:
        Api: An instance of the PyYoutube Api class.

    Raises:
        Exception: If the client secret file is not found or contains errors.
    """
    try:
        client_secret_path = os.path.join(os.getcwd(), "channels", channel_name, "credentials", "client_secret.json")
        with open(client_secret_path, "r") as file:
            client_secret_data = json.load(file)
            client_id = client_secret_data['installed']['client_id']
            client_secret = client_secret_data['installed']['client_secret']

        logger.info(f"Successfully loaded client secrets for channel: {channel_name}")
        return Api(client_id=client_id, client_secret=client_secret)
    except FileNotFoundError:
        logger.error(f"Client secret file not found for channel: {channel_name}")
        raise Exception(f"Google Cloud client secrets not found for channel {channel_name}")
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in client secret file for channel: {channel_name}")
        raise Exception(f"Invalid client secret file for channel {channel_name}")
    except KeyError as e:
        logger.error(f"Missing key in client secret file: {e}")
        raise Exception(f"Invalid client secret file structure for channel {channel_name}")

def clear_token(channel_name: str):
    access_token_path = os.path.join(os.getcwd(), "channels", channel_name, "credentials", "access_token.json")
    if os.path.exists(access_token_path):
        # clear the content of the file
        with open(access_token_path, "w") as file:
            file.write("")

def get_access_token(channel_name: str, clear_token_flag: bool = False) -> AccessToken:
    """
    Retrieve or generate an access token for the specified channel.

    Args:
        channel_name (str): The name of the channel to get the access token for.

    Returns:
        Union[None, AccessToken]: An AccessToken instance if successful, None otherwise.
    """
    access_token_path = os.path.join(os.getcwd(), "channels", channel_name, "credentials", "access_token.json")
    if clear_token_flag:
        clear_token(channel_name)
    # Check if access token already exists
    print(f'\n\n {access_token_path=}');
    print('\n ============\n\n');
    try:
        if os.path.exists(access_token_path):
            with open(access_token_path, "r") as file:
                logger.info(f"Loading existing access token for channel: {channel_name}")
                return AccessToken.from_dict(json.load(file))
    except Exception as e:
        logger.error(f"Error reading existing access token file: {str(e)}")
    
    # Get a new authorization URL
    api = get_api(channel_name)
    auth_url, _ = api.get_authorization_url()
    logger.info(f"Generated new authorization URL for channel: {channel_name}")
    print(f"Please visit this URL to authorize your application: {auth_url}")
    
    # After visiting the URL and authorizing, you'll be redirected. Copy the full redirect URL and paste it here
    redirect_response = input("Enter the full redirect URL: ")

    # Generate the access token
    try:
        access_token = api.generate_access_token(authorization_response=redirect_response)
        
        if not isinstance(access_token, AccessToken):
            raise Exception(f"Invalid access token type 82: {type(access_token)}")
        with open(access_token_path, "w") as file:
            json.dump(access_token.to_dict(), file)
        logger.info(f"Successfully generated and saved new access token for channel: {channel_name}")
        return access_token
    except Exception as e:
        logger.error(f"Error generating access token: {str(e)}")
        raise Exception(f"Error generating access token: {str(e)}")

def get_refresh_token(channel_name: str) -> Union[None, str]:
    """
    Refresh the access token for the specified channel.

    Args:
        channel_name (str): The name of the channel to refresh the token for.

    Returns:
        Union[None, str]: The new access token if successful, None otherwise.

    Raises:
        Exception: If the access token is not found for the channel.
    """
    access_token = get_access_token(channel_name)
    if access_token is None:
        logger.error(f"Access token not found for channel: {channel_name}")
        raise Exception(f"Access token not found for channel {channel_name}")
    
    api = get_api(channel_name)
    try:
        refresh_token = api.refresh_token(refresh_token=access_token.refresh_token)
        if isinstance(refresh_token, AccessToken):
            logger.info(f"Successfully refreshed token for channel: {channel_name}")
            file_path = os.path.join(os.getcwd(), "channels", channel_name, "credentials", "access_token.json")
            with open(file_path, "w") as file:
                json.dump(refresh_token.to_dict(), file)
            return refresh_token.access_token
        else:
            logger.error(f"Failed to retrieve refresh token for channel: {channel_name}")
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
    
    return None

# Example usage
# get_access_token('EatonWorld')
# get_refresh_token('hussainwali')
