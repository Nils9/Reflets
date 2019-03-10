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
img2 = cv2.imread('jean-moral//jean-moral_2_p.jpg',0)
N,M = img1.shape

newImage = np.hstack((img1, img2))#hstack et vstack

def disp(im,title="",ratio=2):
    height, width = im.shape[:2]
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, width//ratio, height//ratio)
    cv2.imshow(title, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
disp(newImage,"Deux images",ratio=4)
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
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],matchColor=[0,255,0],outImg=img1, flags=2)

disp(img3)

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
disp(dst)

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
    

def recalage(filename,I,base_nb=1,show=False,ratio=2):
    L_rec=[]
    img_base = cv2.imread(filename+'//'+filename+"_"+str(base_nb)+".jpg",0)
    for i in I:
        file=filename+'//'+filename+"_"+str(i)+".jpg"
        img = cv2.imread(file,0)
        l_k=find_key_points(img_base,img)
        H = cv2.findHomography(np.float32(l_k[0]),np.float32(l_k[1]),cv2.RANSAC)[0]
        
        H_inv = np.linalg.inv(H)
        rows,cols = img.shape
        dst = cv2.warpPerspective(img,H_inv,(cols,rows))
        if show:
            
            img_f = np.hstack((img,dst,img_base))
            disp(img_f,"Recalage : " + str(i),ratio)
            
        L_rec.append(dst)
    return(L_rec)

I = [1,2,3,4,5,7,8,9,10,11,12,13]
recalage("thamar",I,3,True,10)
