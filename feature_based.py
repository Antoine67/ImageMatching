# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:20:54 2021

@author: Antoine
"""
import cv2 as cv
import csv


files = ["graf1.png"]




sift = cv.xfeatures2d_SIFT.create()
surf = cv.xfeatures2d_SURF.create()
orb = orb = cv.ORB_create()

feature_based = {"SIFT":sift, "SURF":surf,"ORB":orb}
out_csv_dict = {}

for file in files:
    
    image = cv.imread(file)
    gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    out_csv_dict[file] = {} 
    
    for key_f_b in feature_based:
        
        f_b = feature_based[key_f_b]
        img_copy = image
        keyPoints = f_b.detect(img_copy,None)
        output = cv.drawKeypoints(img_copy,keyPoints,None)
        
        
        out_csv_dict[file][key_f_b] = {}
        out_csv_dict[file][key_f_b]['nb_keypoints']  = len(keyPoints)
        
        #cv.imshow("FEATURES KEYPOINTS - " + str(f_b),output)
        
    


with open('results/output-feature-based.csv', 'w+', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for key in out_csv_dict: # filenames
        for key2 in out_csv_dict[key]: # methods (SURF, SIFT,..)
            filewriter.writerow([str(key), str(key2), str(out_csv_dict[key][key2]['nb_keypoints'])])
           #print(str(out_csv_dict[key][key2]['nb_keypoints']))



#cv.imshow("NORMAL",image)

#cv.waitKey(0)
#cv.destroyAllWindows()
