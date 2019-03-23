#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:27:30 2019

@author: nils
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2
from scipy import ndimage
import os
from PIL import Image


#%% Fonction d'affichage d'une image

def disp(im,title="",ratio=2):
    height, width = im.shape[0], im.shape[1]
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, width//ratio, height//ratio)
    cv2.imshow(title, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Test 1 : affichage de deux images côte à côte
img1 = cv2.imread('thamar//thamar_1.jpg', 1)
print(img1.shape)
img2 = cv2.imread('jason//jason_2.jpg', 1)
#newImage = np.hstack((img1, img2))#hstack et vstack
#disp(newImage,"Deux images",ratio=4)

#%% Trouver les points clés de deux images avec les ORB

def find_key_points(img1, img2):
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
    
#    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], matchColor=[0,255,0], outImg=img1, flags=2)
#    disp(img3) 
    
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

#Test 2 : afficher les correspondances entre points clés (décommenter les lignes 50/51)
#find_key_points(img1, img2)
    
#%% Création d'un damier entre deux images pour observer le recalage

def damier(im1, im2, resolution):
    (n,m) = (im1.shape[0], im1.shape[1])
    (n1, m1) = (im2.shape[0], im2.shape[1])
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
        
        #disp(newIm)

    else:
        print("erreur")
        
    return newIm

#Test 3 : afficher deux images en damier (décommenter la ligne 98)
#damier(img1, img2, 16)
    
#%% Fonction de recalage des images
    
def recalage(filename, t, show, show_compare, show_damier, ratio=2): 
#t = 0 si niveaux de gris, 1 si en couleur
        
    nbImages = len(os.listdir(filename))
    L_rec=[]
    img_base = cv2.imread(filename+'//'+filename+"_1.jpg", t)
    
    for i in range(nbImages):
        file = filename + '//' + filename + "_" + str(i+1) + ".jpg"
        img = cv2.imread(file, t)
        l_k = find_key_points(img_base, img)
        H = cv2.findHomography(np.float32(l_k[0]), np.float32(l_k[1]), cv2.RANSAC)[0]
        H_inv = np.linalg.inv(H)
        rows, cols = img.shape[0], img.shape[1]
        dst = cv2.warpPerspective(img, H_inv, (cols,rows))
        
        if show:
    
            img_f = np.hstack((img,dst, img_base))
            disp(img_f, "Recalage : " + str(i), ratio)
            
        if show_compare:
            
            img_fus = cv2.addWeighted(img_base, 0.3, dst, 0.7,0)
            disp(img_fus,"Recalage : " + str(i),ratio)
        
        if show_damier:
            disp(damier(img_base, dst, 16))
            
        L_rec.append(dst)
        
    return(L_rec)

#Test 3 : recalage des images d'un fichier sur la première image
#filename = "jean-moral"
#recalage(filename, 1, False, False, True, 2)
    
#%% Suppression du reflet
#Codes: 
#1 : méthode de la plus faible valeur
#2 : méthode de la moyenne
#3 : méthode de la moyenne des 2 meilleurs résultats

def reflects(filename, code, t):
    recal = recalage(filename, t, False, False, False, 2)
    (n,m) = (recal[0].shape[0], recal[0].shape[1])
    img = recal[0] #image de sortie
    imCompare = recal[0] #image de comparaison (NB)
    if(t):
       imCompare = cv2.cvtColor(recal[0], cv2.COLOR_BGR2GRAY)       
        
    if(code == 1):
        for i in range(1, len(recal)):
            print("Traitement de l'image " + str(i))
            if(t):
                currentIm = cv2.cvtColor(recal[i], cv2.COLOR_BGR2GRAY)            
            else:
                currentIm = recal[i]
                
            for j in range(n):
                for k in range(m):
                    if (imCompare[j][k] > currentIm[j][k]):
                        imCompare[j][k] = currentIm[j][k]
                        img[j][k] = recal[i][j][k]
        
#    elif(code == 2):
#        for i in range(1, len(recal)):
#            print(i)
#            for j in range(n):
#                for k in range(m):
#                    newImg[j][k] += recal[i][j][k]
#        for i in range(n):
#            for j in range(m):
#                newImg[i][j] = newImg[i][j]/len(recal)
#    
#    elif(code == 3):
#        counter = [[0] * m for _ in range(n)]
#        for i in range(1, len(recal)):
#            for j in range(n):
#                for k in range(m):
#                    if (newImg[j][k] > recal[i][j][k]):
#                        newImg[j][k] = recal[i][j][k]
#                        counter[j][k] = [i]
#                        
#        image2 = recal[0]
#        for i in range(1, len(recal)):
#           for j in range(n):
#               for k in range(m):
#                   if (image2[j][k] > recal[i][j][k] and counter[j][k] != i):
#                       image2[j][k] = recal[i][j][k]
#        newImg = (image2 + newImg)/2
        
        
    
#    img = Image.fromarray(newImg, 'L')
#    img.show()
    disp(img)
    
    return img

newImg = reflects("galatee", 1, 1)
#im = Image.fromarray(newImg, 'L')
#im.save("thamar.jpeg")