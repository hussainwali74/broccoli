from together import Together
import base64
from dotenv import load_dotenv
import os

load_dotenv()


def create_from_data(data: list) -> None:
    """
    Generate images from the provided data and save them to the 'images' directory.

    Args:
        data (list): A list of dictionaries containing image descriptions.

    Returns:
        None

    This function iterates through the provided data, generates images for elements
    of type 'image', and saves them in the 'images' directory. It creates the directory
    if it doesn't exist.
    """
    
    if not os.path.exists('images'):
        os.makedirs('images')
    image_number = 0
    for i, element in enumerate(data):
        if element['type'] != 'image':
            continue
        image_number += 1
        generate_image(element['description'], os.path.join('images', f'image_{image_number}.webp'))
        print(f'generating image {image_number} done')


def together_image_gen(prompt, output_file, size, steps=4):
    """
    Generate an image using the Together AI API.

    Args:
        prompt (str): The text prompt describing the desired image.
        output_file (str): Path where the generated image will be saved.
        size (str): Image dimensions in format 'widthxheight' (e.g. '1024x1792').
        steps (int, optional): Number of diffusion steps. More steps generally means higher quality
                             but slower generation. Defaults to 4.

    Returns:
        Dict: The raw API response containing generation metadata.

    The function uses the FLUX.1-schnell-Free model from black-forest-labs to generate the image.
    The generated image is saved in base64 format and then decoded to a binary file at the specified
    output path.
    """
    client = Together()
    try:
        print(f'\n\n {prompt=}')
        print('\n ============\n\n')
        response = client.images.generate(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell-Free",
            width=size.split('x')[0],
            height=size.split('x')[1],
            steps=steps,
            n=1,
            response_format="b64_json"  
        )
        img_b64 = response.data[0].b64_json
        with open(output_file, "wb") as file:
            file.write(base64.b64decode(img_b64))
        return response
    except Exception as e:
        print(f"\n\n Error generating image imagen 71: {str(e)}")
        return {"error": str(e)}

def generate_image(prompt, output_file, size='1024x1792'):
    """
    Generate an image using the Together AI API.

    Args:
        prompt (str): The text prompt describing the desired image.
        output_file (str): Path where the generated image will be saved.
        size (str, optional): Image dimensions in format 'widthxheight'. Defaults to '1024x1792'.

    Returns:
        Dict: The raw API response containing generation metadata.

    This is a wrapper function that calls together_image_gen() with the provided parameters.
    It simplifies the image generation process by providing default values and handling
    the API interaction through the Together AI client.
    """
    response = together_image_gen(prompt, output_file, size)
    return response
# ================================================================================================== PARALLEL IMAGE GEN
# cant use because of rate limit per minute 6 requests allowed
import asyncio
import aiohttp
import base64
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

async def async_together_image_gen(prompt: str, output_file: str, size: str, steps: int = 4) -> Dict:
    """
    Asynchronously generate an image using the Together API.

    Args:
        prompt (str): The image description prompt.
        output_file (str): Path to save the generated image.
        size (str): Image size in format 'widthxheight'.
        steps (int): Number of generation steps (default: 4).

    Returns:
        Dict: The API response.
    """
    from together import Together

    client = Together()

    try:
        width, height = map(int, size.split('x'))
        response = await asyncio.to_thread(
            client.images.generate,
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell-Free",
            width=width,
            height=height,
            steps=steps,
            n=1,
            response_format="b64_json"
        )

        img_b64 = response.data[0].b64_json
        with open(output_file, "wb") as file:
            file.write(base64.b64decode(img_b64))
        return response
    except Exception as e:
        print(f"Error generating image for prompt '{prompt}': {str(e)}")
        return None

async def generate_images_parallel(prompts: List[str], output_files: List[str], size: str = '1024x1792', steps: int = 4) -> List[Dict]:
    """
    Generate multiple images in parallel using the Together API.

    Args:
        prompts (List[str]): List of image description prompts.
        output_files (List[str]): List of output file paths.
        size (str): Image size in format 'widthxheight' (default: '1024x1792').
        steps (int): Number of generation steps (default: 4).

    Returns:
        List[Dict]: List of API responses.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [
            async_together_image_gen(prompt, output_file, size, steps)
            for prompt, output_file in zip(prompts, output_files)
        ]
        responses = await asyncio.gather(*tasks)
    return responses

def generate_images(prompts: List[str], output_files: List[str], size: str = '1024x1792', steps: int = 4) -> List[Dict]:
    """
    Generate multiple images using the Together API.

    Args:
        prompts (List[str]): List of image description prompts.
        output_files (List[str]): List of output file paths.
        size (str): Image size in format 'widthxheight' (default: '1024x1792').
        steps (int): Number of generation steps (default: 4).

    Returns:
        List[Dict]: List of API responses.
    """
    return asyncio.run(generate_images_parallel(prompts, output_files, size, steps))

# Example usage:
# prompts = ["A beautiful sunset", "A majestic mountain", "A serene lake"]
# output_files = ["sunset.webp", "mountain.webp", "lake.webp"]
# responses = generate_images(prompts, output_files)
# end \================================================================================================== PARALLEL IMAGE GEN