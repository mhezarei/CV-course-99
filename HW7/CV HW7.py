#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pylab as plt
plt.rcParams["figure.figsize"] = (12,6)


# In[2]:


def show_image(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()


# In[150]:


image = cv2.imread('temp1.jpg', 0)
show_image(image)


# In[151]:


image.shape


# In[166]:


import math

def gaussian_noise(image, mean=10, std=0.0):
    noise = np.zeros((image.shape), dtype=np.uint8)
    cv2.randn(noise, 10, np.std(image))
    return cv2.add(image, noise)


# In[167]:


noisy = gaussian_noise(image)
show_image(noisy)


# In[168]:


show_image(cv2.medianBlur(noisy, 3))


# In[169]:


show_image(cv2.GaussianBlur(noisy, (5, 5), 0))


# In[ ]:





# In[ ]:




