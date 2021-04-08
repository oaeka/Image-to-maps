from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from PIL import Image
import numpy as np


def load_image(path, src=(256, 512)):
    src_list, tar_list = list(), list()

    for filename in listdir(path):
        pixels = load_img(path + filename, target_size=src)
        pixels = img_to_array(pixels)

        sat_img, map_img = pixels[:, 0:256], pixels[:, 256:]
        src_list.append(sat_img)
        tar_list.append(map_img)

    return [asarray(src_list), asarray(tar_list)]


[src_image, tar_image] = load_image('./dataset/maps/train/')
print("loaded images : ", src_image.shape, tar_image.shape)

filename = 'maps_256.npz'
savez_compressed(filename, src_image, tar_image)
print("Saved dataset: ", filename)
