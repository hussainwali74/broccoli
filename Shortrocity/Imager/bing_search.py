from google_images_search import GoogleImagesSearch
import os
import time
# you can provide API key and CX using arguments,
# or you can set environment variables: GCS_DEVELOPER_KEY, GCS_CX
gis = GoogleImagesSearch(developer_key='AIzaSyBLrMwHNDVVVBhJyJpvn65x0SCqqfkJm_o',custom_search_cx="25be1bdb8030b4d90")

# define search params
# option for commonly used search param are shown below for easy reference.
# For param marked with '##':
#   - Multiselect is currently not feasible. Choose ONE option only
#   - This param can also be omitted from _search_params if you do not wish to define any value
_search_params = {
    'q': 'macbook 15 pro max',
    'num': 2,
    'fileType': 'jpg',
    'rights': 'cc_publicdomain',
    'safe': 'high', ##
    'imgType': 'photo', ##
    'imgSize': 'xlarge', ##
    'imgDominantColor': 'imgDominantColorUndefined' 
}
# Attempt to search for images and download them
while True:
   try:
       # Perform the search
       gis.search(search_params=_search_params, path_to_dir='./image_search_result/')
       
       # Print the results
       print("-------------------------------------------------------------------------")
       print(f"{gis.results()=}")
       print("-------------------------------------------------------------------------")

       # Download and resize the images
       i = 0
       for image in gis.results():
           print("-------------------------------------------------------------------------")
           print(f"{image.path=}")
           print("-------------------------------------------------------------------------")
           old_path = image.path
           new_path = os.path.join('./image_search_result/', 'new_filename'+str(i)+'.jpg') # Specify your new filename here
           os.rename(old_path, new_path)
           image.download('./image_search_result/')
        #    image.resize(500, 500)
           

       break # Break the loop if the request is successful
   except Exception as e:
       print(f"An error occurred: {e}. Retrying in 1 second...")
       time.sleep(1) # Wait for 5 seconds before retrying