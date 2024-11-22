import os
import urllib.request
from tqdm import tqdm

def download_sam_model():
    """
    Download the SAM model checkpoint file using urllib.
    """
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # URL for the model
    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    
    # Local path to save the model
    save_path = os.path.join(models_dir, "sam2_hiera_large.pt")
    
    # Check if model already exists
    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}")
        return save_path
    
    print(f"Downloading SAM model to {save_path}...")
    
    # Create a class to handle the progress bar
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    # Download with progress bar
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=save_path,
                                 reporthook=t.update_to)
    
    print("Download completed!")
    print(f"Model saved to: {save_path}")
    return save_path

if __name__ == "__main__":
    model_path = download_sam_model()
