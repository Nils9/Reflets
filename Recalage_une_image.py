# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:12:09 2019

@author: Simon
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy import ndimage

#%% Récupérer l'image de base et l'image à recaler
img1 = cv2.imread('jean-moral//jean-moral_6_p.jpg',0)
img2 = cv2.imread('jean-moral//jean-moral_2_p.jpg',0)#9
N,M = img1.shape
plt.imshow(img1,cmap='gray')
plt.show()
plt.imshow(img2,cmap='gray')
plt.show()
#%% Identifier les keypoints et les matcher

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)[:100]
kp2, des2 = orb.detectAndCompute(img2,None)[:100]

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors (on regarde les props numériques / descriptors pour matcher les points)
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:4],matchColor=[0,255,0],outImg=img1, flags=2)

plt.figure(figsize=(10,10))
plt.imshow(img3)
plt.show()

# Initialize lists
list_kp1 = []
list_kp2 = []

for mat in matches:

    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
    [x1,y1] = kp1[img1_idx].pt
    [x2,y2] = kp2[img2_idx].pt

    # Append to each list
    list_kp1.append([x1, y1])
    list_kp2.append([x2, y2])
    #Bizarre : pas des integers

#%% Déduire l'homographie en utilisant les keypoints, calculer l'image inverse
H = cv2.findHomography(np.float32(list_kp1),np.float32(list_kp2),cv2.RANSAC)[0]

H_inv = np.linalg.inv(H)

rows,cols = img2.shape
dst = cv2.warpPerspective(img2,H_inv,(cols,rows))
plt.imshow(dst,cmap='gray')

#%% On les recalle toutes
#file : nom du grand fichier et des minis

def find_key_points(img1,img2):
    # Initiate SIFT detector
    orb = cv2.ORB_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)[:100]
    kp2, des2 = orb.detectAndCompute(img2,None)[:100]
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors (on regarde les props numériques / descriptors pour matcher les points)
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Initialize lists
    list_kp1 = []
    list_kp2 = []
    
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        
        # x - columns
        # y - rows
        # Get the coordinates
        [x1,y1] = kp1[img1_idx].pt
        [x2,y2] = kp2[img2_idx].pt

        # Append to each list
        list_kp1.append([x1, y1])
        list_kp2.append([x2, y2])
    
    return([list_kp1,list_kp2])
    
I = [2,3,4,5,7,8,9]

def recalage(show=False):
    L_rec=[]
    img_base = cv2.imread('jean-moral//jean-moral_6_p.jpg',0)
    for i in I:
        filename='jean-moral//jean-moral_'+str(i)+'_p.jpg'
        img = cv2.imread(filename,0)
        l_k=find_key_points(img_base,img)
        H = cv2.findHomography(np.float32(l_k[0]),np.float32(l_k[1]),cv2.RANSAC)[0]
        
        H_inv = np.linalg.inv(H)
        rows,cols = img.shape
        dst = cv2.warpPerspective(img,H_inv,(cols,rows))
        if show:
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
            ax1.imshow(img,cmap='gray')
            ax1.set_title('Before')
            ax2.imshow(dst,cmap='gray')
            ax2.set_title('After')
            ax3.imshow(img_base,cmap='gray')
            ax3.set_title('Base')
            plt.show()
        L_rec.append(dst)
    return(L_rec)
        
recalage(True)
        