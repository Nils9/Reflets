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
from PIL import Image

img1 = 'jean-moral//jean-moral_6_p.jpg'
img2 = 'jean-moral//jean-moral_5_p.jpg'

def displayWithTransparency(img2, img1, alpha):
    foreground = Image.open(img1)   
    background = Image.open(img2)

    foreground.putalpha(alpha) #ajout de la transparence
    
    background.paste(foreground, (0, 0), foreground)
    background.show()


displayWithTransparency(img2, img1, 50)

