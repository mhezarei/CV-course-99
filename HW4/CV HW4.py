#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import matplotlib.pylab as plt
plt.rcParams["figure.figsize"] = (12,6)

def show_image(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

image = cv2.imread('temp.jpg', 0) # reading image in grayscale

show_image(image)

# histogram of the image
plt.plot(cv2.calcHist([image], [0], None, [256], [0, 256]))
plt.show()

def enhance(image):
    temp = np.zeros((image.shape), dtype='uint8')
    minn = image.min()
    maxx = image.max()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp[i, j] = int(255 * (image[i,j] - minn) / (maxx - minn))
    return temp

new_image = enhance(image)
show_image(new_image)

histo = {}
for i in range(new_image.shape[0]):
    for j in range(new_image.shape[1]):
        if new_image[i][j] not in histo:
            histo[new_image[i][j]] = 1
        else:
            histo[new_image[i][j]] += 1

x, y = zip(*sorted(histo.items()))
plt.plot(x, y)
plt.show()
