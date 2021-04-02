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
import math 
import imutils
import pandas as pd

method_array = {}
method_array['rotation'] = True
method_array['zoom'] = True
method_array['blur'] = True
method_array['noise'] = True


sizes = [128,256]

df_sizes = {128 : pd.read_csv("../results/template_creation_128.csv",header=0, dtype=object,sep=';') ,
            256 : pd.read_csv("../results/template_creation_256.csv",header=0, dtype=object,sep=';')}

folder_name = "dataset1" #dataset folder name

def createFolder(strpath):
   if(not path.isdir(strpath)):
    try:
        os.mkdir(strpath)
    except OSError:
        print ("Creation of the directory failed" + strpath)
    else:
        print ("Successfully created the directory" + strpath)
        
        


        
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


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = ((w-1) // 2.0, (h-1)// 2.0)


    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    print (nW, nH)
    
    # adjust the rotation matrix to take into account translation
    M[0, 2] += ((nW-1) / 2.0) - cX
    M[1, 2] += ((nH-1) / 2.0) - cY
    
    # perform the actual rotation and return the image
    return M, cv.warpAffine(image, M, (nW, nH))

#function that calculates the updated locations of the coordinates
#after rotation
def rotated_coord(points,M):
    points = np.array(points)
    ones = np.ones(shape=(len(points),1))
    points_ones = np.concatenate((points,ones), axis=1)
    transformed_pts = M.dot(points_ones.T).T
    return transformed_pts


createFolder("../storage/dataset_alter/" + folder_name)


for file in os.listdir("../storage/dataset/" + folder_name):
    if file.endswith(".png") or file.endswith(".jpeg"):
        
        image = cv.imread("../storage/dataset/" + folder_name + "/" + str(file))
        
        
        if(method_array['noise']):
            #noise
            createFolder("../storage/dataset_alter/" + folder_name + "/noise")
            
            noise = add_noise(image, 3)
            
            cv.imwrite("../storage/dataset_alter/" + folder_name + "/noise/" + str(file), noise)
            
           
                
        
        if(method_array['rotation']):
            #rotation
            createFolder("../storage/dataset_alter/" + folder_name + "/rotation")
            
            h,w,c = image.shape
            angle = 45

            M, rotation = rotate_bound(image, angle )
            
           
            
            cv.imwrite("../storage/dataset_alter/" + folder_name + "/rotation/" + str(file), rotation) 
            for s in sizes:
                
                df_sizes[s]['x1'] = df_sizes[s]['x1'].astype(int)
                df_sizes[s]['y1'] = df_sizes[s]['y1'].astype(int)
                df_sizes[s]['x2'] = df_sizes[s]['x2'].astype(int)
                df_sizes[s]['y2'] = df_sizes[s]['y2'].astype(int)
                
                
                
                for idx,row in df_sizes[s].loc[df_sizes[s]['image'] == file].iterrows():
                    
                    res = rotated_coord(np.array([[row['x1'], row['y1']]]),M)
                    
                    df_sizes[s].at[idx,'x1_rotation'] = res[0][0]
                    df_sizes[s].at[idx,'y1_rotation'] = res[0][1]
                    
                    
                    res = rotated_coord(np.array([[row['x2'], row['y2']]]),M)
                    df_sizes[s].at[idx,'x2_rotation'] = res[0][0]
                    df_sizes[s].at[idx,'y2_rotation'] = res[0][1]
                
           
        
        if(method_array['zoom']):
            #zoom
            createFolder("../storage/dataset_alter/" + folder_name + "/zoom")
    
            fx = 0.5
            fy = 0.5
            zoom = cv.resize(image,None,fx=fx, fy=fy, interpolation = cv.INTER_CUBIC)
             
            cv.imwrite("../storage/dataset_alter/" + folder_name + "/zoom/" + str(file), zoom)
            
            for s in sizes:
                
                df_sizes[s]['x1'] = df_sizes[s]['x1'].astype(int)
                df_sizes[s]['y1'] = df_sizes[s]['y1'].astype(int)
                df_sizes[s]['x2'] = df_sizes[s]['x2'].astype(int)
                df_sizes[s]['y2'] = df_sizes[s]['y2'].astype(int)
                
            
                
                
                for idx,row in df_sizes[s].loc[df_sizes[s]['image'] == file].iterrows():
                                        
                    df_sizes[s].at[idx,'x1_zoom'] = row['x1'] * fx
                    df_sizes[s].at[idx,'y1_zoom'] = row['y1'] * fy
                    
                    
                    df_sizes[s].at[idx,'x2_zoom'] = row['x2'] * fx
                    df_sizes[s].at[idx,'y2_zoom'] = row['y2'] * fy
        
        
        if(method_array['blur']):
            #blur
            createFolder("../storage/dataset_alter/" + folder_name + "/blur")
            
            blur = cv.blur(image,(10,10))  
            
            cv.imwrite("../storage/dataset_alter/" + folder_name + "/blur/" + str(file), blur)


for s in sizes:
     df_sizes[s].to_csv(f'../results/template_creation_after_pipeline_{s}.csv',sep=';',index=False)