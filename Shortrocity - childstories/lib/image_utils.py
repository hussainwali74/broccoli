import cv2
import numpy as np

def resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resize an image to fit within specified dimensions while maintaining aspect ratio.

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        width (int): The maximum width of the resized image.
        height (int): The maximum height of the resized image.

    Returns:
        numpy.ndarray: The resized image as a NumPy array.

    This function resizes the input image to fit within the specified width and height
    while maintaining its original aspect ratio. If the image's aspect ratio is wider
    than the target dimensions, it will be resized to the full width. Otherwise, it
    will be resized to the full height.
    """
    # calculate the aspect ratio of the original image
    aspect_ratio = image.shape[1] / image.shape[0]
    
    # calculate the new dimensions to fit within the width and height while maintaining the aspect ratio
    if aspect_ratio > (width/height):
        new_width = width
        new_height = int(width / aspect_ratio)
    else:
        new_height = height
        new_width = int(height * aspect_ratio)
    
    # resize the image to the new dimensions
    return cv2.resize(image, (new_width, new_height))
