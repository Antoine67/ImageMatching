# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:20:54 2021

@author: Antoine
"""
import cv2 as cv
import csv
import os
from SIFT import SIFTMethod
from TMCoeffNormed import TMCoeffNormedMethod


img_folder_name = "dataset1"
temp_folder_name = "dataset1_templates/128"

img_files = {}
img_and_temp_file = {}

#
for file in os.listdir("../storage/dataset/" + img_folder_name):
    if file.endswith(".png") or file.endswith(".jpeg"):
        img_and_temp_file[file] = []
        
# Add associated template        
for file in os.listdir("../storage/dataset/" + temp_folder_name):
    if file.endswith(".png") or file.endswith(".jpeg"):
        for img_file in img_and_temp_file:
            file_name_only = os.path.basename(img_file)[:-4]
            if(file.startswith(file_name_only + "_")):
                img_and_temp_file[img_file].append(file)
                
#Affiche les images et templates associés             
#print(img_and_temp_file)



s =  SIFTMethod()
tm_coeff_normed = TMCoeffNormedMethod()


feature_based = [s, tm_coeff_normed]
out_csv_dict = {}

for img_path_name in img_and_temp_file:
    for temp_path_name in img_and_temp_file[img_path_name]:
        
        img_path = "../storage/dataset/" + img_folder_name +'/'+ img_path_name
        temp_path = "../storage/dataset/" + temp_folder_name +'/'+  temp_path_name
        
        print('Matching files : img -> ' + img_path + ' & template -> ' + temp_path)
    
        out_csv_dict[img_path + ' ' + temp_path ] = {} 
        
        for f_b in feature_based:
            if(f_b.grayscale):
                img_full = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
                img_temp = cv.imread(temp_path,cv.IMREAD_GRAYSCALE)
            else:
                img_full = cv.imread(img_path,0)
                img_temp = cv.imread(temp_path,0)
                
            f_b.set_pictures(img_full, img_temp)
            key_f_b = f_b.name 
            
            matches, execution_time = f_b.match()
            
            out_csv_dict[img_path + ' ' + temp_path ][key_f_b] = {}
            out_csv_dict[img_path + ' ' + temp_path ][key_f_b]['nb_matches']  = matches
            out_csv_dict[img_path + ' ' + temp_path ][key_f_b]['execution_time']  = execution_time
               
    


with open('../results/output.csv', 'w+', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    filewriter.writerow([
                "Picture",
                "Method",
                "Match count",
                "Execution time (s)"
            ])
    
    for key in out_csv_dict: # filenames
        for key2 in out_csv_dict[key]: # methods (SURF, SIFT,..)
            filewriter.writerow([
                str(key),
                str(key2),
                str(out_csv_dict[key][key2]['nb_matches']),
                str(out_csv_dict[key][key2]['execution_time'])
            ])



#cv.imshow("NORMAL",image)

#cv.waitKey(0)
#cv.destroyAllWindows()
