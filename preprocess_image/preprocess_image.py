import numpy as np
import PIL.ImageOps as ImageOps
from bin_module.bin import bradley_binariation
from PIL import Image

def rec_digit(img):
  
    img = img.convert("L")

    pixels_list = list(img.getdata())

    bradley_binariation(pixels_list, img.width, img.height, 4)

    two_dimensional = np.array([pixels_list[i:i+img.width] for i in range(0, len(pixels_list), img.width)])

    while np.sum(two_dimensional[0]) // len(two_dimensional[0]) == 255:
        two_dimensional = two_dimensional[1:]
    while np.sum(two_dimensional[:,0]) // len(two_dimensional[:,0]) == 255:
        two_dimensional = np.delete(two_dimensional,0,1)
    while np.sum(two_dimensional[-1]) // len(two_dimensional[-1]) == 255:
        two_dimensional = two_dimensional[:-1]
    while np.sum(two_dimensional[:,-1]) // len(two_dimensional[:,-1]) == 255:
        two_dimensional = np.delete(two_dimensional,-1,1)

    # a = rescaled_matrix((20, 20), two_dimensional)
    # a = a.flatten().tolist()
    # bin.anti_aliasing(a, 20, 20, 1)
    # new_img = Image.new(mode="L", size=(20, 20))
    # new_img.putdata(a)
    new_img = Image.new(mode="L", size=tuple(two_dimensional.shape[::-1]))
    new_img.putdata(two_dimensional.flatten())

    new_img = new_img.resize((20, 20))
    new_img = ImageOps.invert(new_img)
    resized_array = np.array(new_img)
    resized_array = np.lib.pad(resized_array, (4, 4), 'constant')
    new_img = new_img.resize((28, 28))
    new_img.putdata(resized_array.flatten())

    return new_img


def rescaled_matrix(desired_shape, original_array):
    original_shape = original_array.shape

    stretch_factors = (desired_shape[0] / original_shape[0], desired_shape[1] / original_shape[1])

    resized_array = np.zeros(desired_shape, dtype=original_array.dtype)
    for i in range(desired_shape[0]):
        for j in range(desired_shape[1]):
            x = i / stretch_factors[0]
            y = j / stretch_factors[1]
            resized_array[i, j] = original_array[int(x), int(y)]
    
    return resized_array
