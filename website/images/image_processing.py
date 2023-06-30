import numpy as np
import cv2
from PIL import Image
import skimage


def closest_power_of_two(img:np.array):

    shortest_side=np.min(img.shape)

    return 2**int(np.floor(np.log2(shortest_side)))

def img_to_binary_array(path_input:str):
    img = cv2.imread(path_input, 2)

    shape=(closest_power_of_two(img),closest_power_of_two(img))

    img = cv2.resize(img, shape, interpolation = cv2.INTER_AREA)

    _, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)

    return img//255

def save_img_to_bmp(img:np.array, path_output:str):
    out_img = Image.new('1', img.shape)
    pixels = out_img.load()

    for i in range(out_img.size[0]):    
        for j in range(out_img.size[1]):
            pixels[i, j] = int(img[j, i])

    out_img.save(path_output)


def img_to_bmp(path_input, path_output, resize=1):

    img=img_to_binary_array(path_input)

    img=skimage.measure.block_reduce(img, (resize, resize), np.min)

    save_img_to_bmp(img,path_output)
