# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:20:54 2021

@author: Nicolas
"""
import cv2 as cv
import csv
import os, os.path, time
import numpy as np
import matplotlib.pyplot as plt
from os import path

method_array = {}
method_array['rotation'] = True
method_array['zoom'] = True
method_array['blur'] = True
method_array['noise'] = True

folder_name = "test"

def createFolder(strpath):
   if(not path.isdir(strpath)):
    try:
        os.mkdir(strpath)
    except OSError:
        print ("Creation of the directory failed")
    else:
        print ("Successfully created the directory")
        
def add_noise(X_img, number):
    img = X_img[...,::-1]/255.0
    noise =  np.random.normal(loc=0, scale=1, size=img.shape)
    
    # noise overlaid over image
    noisy = np.clip((img + noise*0.2),0,1)
    noisy2 = np.clip((img + noise*0.4),0,1)
    
    # noise multiplied by image:
    # whites can go to black but blacks cannot go to white
    noisy2mul = np.clip((img*(1 + noise*0.2)),0,1)
    noisy4mul = np.clip((img*(1 + noise*0.4)),0,1)
    
    
    # noise multiplied by bottom and top half images,
    # whites stay white blacks black, noise is added to center
    img2 = img*2
    n2 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.2)), (1-img2+1)*(1 + noise*0.2)*-1 + 2)/2, 0,1)
    n4 = np.clip(np.where(img2 <= 1, (img2*(1 + noise*0.4)), (1-img2+1)*(1 + noise*0.4)*-1 + 2)/2, 0,1)
    
    # norm noise for viz only
    noise2 = (noise - noise.min())/(noise.max()-noise.min())
    
    noise_array = {}
    noise_array[0] = noisy
    noise_array[1] = noisy2
    noise_array[2] = noisy2mul
    noise_array[3] = noisy4mul
    noise_array[4] = n2
    noise_array[5] = n4

    return_image = noise_array[number]*255
    return_image = return_image.astype(np.uint8) 
    return return_image 

createFolder("storage/dataset_alter/" + folder_name)

for file in os.listdir("storage/dataset/" + folder_name):
    if file.endswith(".png") or file.endswith(".jpeg"):
        
        image = cv.imread("storage/dataset/" + folder_name + "/" + str(file))
        
        
        if(method_array['noise']):
            #noise
            createFolder("storage/dataset_alter/" + folder_name + "/noise")
            
            noise = add_noise(image, 3)
            
            cv.imwrite("storage/dataset_alter/" + folder_name + "/noise/" + str(file), noise)
        
        
        if(method_array['rotation']):
            #rotation
            createFolder("storage/dataset_alter/" + folder_name + "/rotation")
            
            rows,cols,etc = image.shape
            M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),45,1)
            rotation = cv.warpAffine(image,M,(cols,rows))
            
            cv.imwrite("storage/dataset_alter/" + folder_name + "/rotation/" + str(file), rotation)
        
        
        if(method_array['zoom']):
            #zoom
            createFolder("storage/dataset_alter/" + folder_name + "/zoom")
    
            zoom = cv.resize(image,None,fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC)
            cv.imwrite("storage/dataset_alter/" + folder_name + "/zoom/" + str(file), zoom)
        
        
        if(method_array['blur']):
            #blur
            createFolder("storage/dataset_alter/" + folder_name + "/blur")
            
            blur = cv.blur(image,(10,10))        
            cv.imwrite("storage/dataset_alter/" + folder_name + "/blur/" + str(file), blur)

