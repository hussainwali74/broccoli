this will be a package for uploading videos to youtube [for insta, tiktok, etc later]
first get access token
then use that access token to upload video

channel name is the identifier.
channels/channel_name/videos will contain the videos to be uploaded, once uploaded move the video to channels/channel_name/videos_uploaded
credentials are stored in channels/channel_name/credentials

Done:
    - Access token 
Todos:
    - add scheduler to upload videos at a specific time
    - add error handling
    - add logging
    - add tests

--------------------------------
Folder Structure:
channels/channel_name/credentials/client_secret.json
channels/channel_name/credentials/access_token.json
channels/channel_name/videos/video_title/video_data.json
channels/channel_name/videos/video_title/video_file_name.{mp4, avi, etc}

channels/channel_name/videos_uploaded/
--------------------------------

step 1:
create gmail account

step 2:
create google cloud project

step 3:
Enable Youtube Data API v3

step 4:
    create oauth client id
    oauth consent screen
    fill details
    
    step 4.1:
        user type: external
    step 4.2:
        add scope: Youtube Data API v3 select all
    step 4.3:
        test users: add your gmail account
    step 4.4:
        save

step 5:
    get access token
        step 5.1: 
            go to Credentials in google cloud project
        step 5.2:
            Create Credentials
        step 5.3:
            select OAuth client ID
        step 5.4:
            select Desktop app
        step 5.5:
            fill details
        step 5.6:
            save


    https://developers.google.com/youtube/v3/docs/videos/insert

