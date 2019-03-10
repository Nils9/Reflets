#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 10:43:20 2019

@author: nils
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.image as mpimg

img1 = Image.open('jean-moral//jean-moral_6_p.jpg')
print(img1.mode)

(n,m) = img1.size

image = img1
image.putalpha(50)

foreground = image

background = Image.open('jean-moral//jean-moral_5_p.jpg')

background.paste(foreground, (0, 0), foreground)
background.show()



