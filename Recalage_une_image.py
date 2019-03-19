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
img1 = cv2.imread('jason//jason_1.jpg',0)
img2 = cv2.imread('jason//jason_2.jpg',0)
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
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],matchColor=[0,255,0],outImg=img1, flags=2)

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

import os

def recalage(filename, show=False,show_compare=False,ratio=2):
    nbImages = len(os.listdir(filename))
    L_rec=[]
    img_base = cv2.imread(filename+'//'+filename+"_1.jpg",0)
    for i in range(nbImages):
        file=filename+'//'+filename+"_"+str(i+1)+".jpg"
        img = cv2.imread(file,0)
        l_k=find_key_points(img_base,img)
        H = cv2.findHomography(np.float32(l_k[0]),np.float32(l_k[1]),cv2.RANSAC)[0]
        
        H_inv = np.linalg.inv(H)
        rows,cols = img.shape
        dst = cv2.warpPerspective(img,H_inv,(cols,rows))
        if show:

            img_f = np.hstack((img,dst,img_base))
            disp(img_f,"Recalage : " + str(i),ratio)
            
        if show_compare:
            
            img_fus = cv2.addWeighted(img_base, 0.3, dst, 0.7,0)
            disp(img_fus,"Recalage : " + str(i),ratio)
            
        L_rec.append(dst)
    return(L_rec)

I = [1,2,3,4,5,7]
recalage("jean-moral",False,True,1)

#%% Superposer deux images avec transparence alpha
from PIL import Image

def displayWithTransparency(background, foreground, alpha):

    foreground.putalpha(alpha) #ajout de la transparence
    
    background.paste(foreground, (0, 0), foreground)
    background.show()


#%% Création d'un damier entre deux images pour observer le recalage

def damier(im1, im2, resolution):
    (n,m) = im1.shape
    (n1, m1) = im2.shape
    if(n==n1 and m==m1):
        newIm = im1
        p = n//resolution
        q = m//resolution
        
        for i in range(resolution):
            for j in range(resolution):
                for k in range(i*p, (i+1)*p):
                    for l in range(j*q, (j+1)*q):
                        if((i+j)%2 == 0):
                            newIm[k][l] = im1[k][l]
                        else:
                            newIm[k][l] = im2[k][l]

        img = Image.fromarray(newIm, 'L')
        img.show()
    else:
        print("erreur")
        
    return newIm

L = recalage("jean-moral", False, False, 2)

im1 = L[0]
im2 = L[1]
damier(im1, im2, 16)
#%% Recherche du reflet
import os
filename = "jean-moral"
nbImages = len(os.listdir(filename))
print(nbImages)

def recalage2(filename, numImg):
    img_base = cv2.imread(filename+'//'+filename+"_1.jpg",0)

    file=filename+'//'+filename+"_"+str(numImg)+".jpg"
    img = cv2.imread(file,0)
    
    l_k=find_key_points(img_base,img)
    H = cv2.findHomography(np.float32(l_k[0]),np.float32(l_k[1]),cv2.RANSAC)[0]
        
    H_inv = np.linalg.inv(H)
    rows,cols = img.shape
    dst = cv2.warpPerspective(img,H_inv,(cols,rows))
    
    for i in range(rows):
        for j in range(cols):
            if(abs(img_base[i][j] - dst[i][j]) > 5):
                if(img_base[i][j] > dst[i][j]):
                    img_base[i][j] = 255
                else:
                    dst[i][j] = 255
    background = Image.fromarray(img_base, 'L')
    background.show()
    
    foreground = Image.fromarray(dst, 'L')
    foreground.show()
                    
recalage2(filename, 2)

#%% Suppression du reflet
#Codes: 
#1 : méthode de la plus faible valeur
#2 : méthode de la moyenne
#3 : méthode de la moyenne des 2 meilleurs résultats

def reflects(filename, code):
    recal = recalage(filename, False, False, 2)
    (n,m) = recal[0].shape
    newImg = recal[0]
    if(code == 1):
        for i in range(1, len(recal)):
            print(i)
            for j in range(n):
                for k in range(m):
                    if (newImg[j][k] < recal[i][j][k]):
                        newImg[j][k] = recal[i][j][k]
        
    elif(code == 2):
        for i in range(1, len(recal)):
            print(i)
            for j in range(n):
                for k in range(m):
                    newImg[j][k] += recal[i][j][k]
        for i in range(n):
            for j in range(m):
                newImg[i][j] = newImg[i][j]/len(recal)
    
    elif(code == 3):
        counter = [[0] * m for _ in range(n)]
        for i in range(1, len(recal)):
            for j in range(n):
                for k in range(m):
                    if (newImg[j][k] > recal[i][j][k]):
                        newImg[j][k] = recal[i][j][k]
                        counter[j][k] = [i]
                        
        image2 = recal[0]
        for i in range(1, len(recal)):
           for j in range(n):
               for k in range(m):
                   if (image2[j][k] > recal[i][j][k] and counter[j][k] != i):
                       image2[j][k] = recal[i][j][k]
        newImg = (image2 + newImg)/2
        
        
    
    img = Image.fromarray(newImg, 'L')
    img.show()
    
    return newImg

newImg = reflects("thamar", 1)
im = Image.fromarray(newImg, 'L')
im.save("thamar.jpeg")

