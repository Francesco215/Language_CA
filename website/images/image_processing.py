import numpy as np
import cv2
from PIL import Image
import skimage
import json
import os


def closest_power_of_two(img: np.array) -> int:
    """
    Calculates the closest power of two for the given image's shortest side.

    Args:
        img (np.array): Input image as a NumPy array.

    Returns:
        int: Closest power of two for the shortest side of the image.
    """
    shortest_side = np.min(img.shape)
    return 2 ** int(np.floor(np.log2(shortest_side)))


def img_to_binary_array(path_input: str, resize=1) -> np.array:
    """
    Converts an image to a binary NumPy array.

    Args:
        path_input (str): Path to the input image file.
        resize (int, optional): Resize factor for the output image. Defaults to 1.

    Returns:
        np.array: Binary NumPy array representing the image.
    """
    img = cv2.imread(path_input, 2)

    shape = (closest_power_of_two(img), closest_power_of_two(img))

    img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)

    _, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

    img = skimage.measure.block_reduce(img, (resize, resize), np.min)

    return img // 255


def save_img_to_bmp(img: np.array, path_output: str):
    """
    Saves a binary NumPy array as a bitmap image.

    Args:
        img (np.array): Binary NumPy array representing the image.
        path_output (str): Path to save the output bitmap image file.
    """
    out_img = Image.new('1', img.shape)
    pixels = out_img.load()

    for i in range(out_img.size[0]):
        for j in range(out_img.size[1]):
            pixels[i, j] = int(img[j, i])

    out_img.save(path_output)


def img_to_bmp(path_input: str, path_output: str, resize: int = 1):
    """
    Converts an image to a binary bitmap and saves it as a bitmap image.

    Args:
        path_input (str): Path to the input image file.
        path_output (str): Path to save the output bitmap image file.
        resize (int, optional): Resize factor for the output image. Defaults to 1.

    """
    img = img_to_binary_array(path_input, resize)

    save_img_to_bmp(img, path_output)


def img_to_json(path_input: str, path_output: str, resize: int = 1):
    """
    Converts an image to a binary NumPy array and saves it as a JSON file.

    Args:
        path_input (str): Path to the input image file.
        path_output (str): Path to save the output JSON file.
        resize (int, optional): Resize factor for the output image. Defaults to 1.
    """
    img = img_to_binary_array(path_input, resize)

    json_str = json.dumps(img.tolist())

    with open(path_output, 'w') as file:
        file.write(json_str)


def get_folder_filenames(folder_path, output_file):
    """
    Retrieve filenames of files in a folder and save them in a JSON file.

    Args:
        folder_path (str): The path of the folder.
        output_file (str): The path of the output JSON file.

    Returns:
        None
    """
    filenames = []

    # Iterate over all items (files and directories) in the folder
    for file_name in os.listdir(folder_path):
        # Check if the item is a file
        if os.path.isfile(os.path.join(folder_path, file_name)):
            # Add the filename to the list
            if file_name!='patterns':
                filenames.append(file_name)

    # Write the filenames list to a JSON file
    with open(output_file, 'w') as f:
        json.dump(filenames, f)
