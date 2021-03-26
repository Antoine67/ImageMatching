# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:20:54 2021

@author: Antoine
"""
import cv2 as cv
import csv
import os
from SIFT import SIFTMethod
from ORB import ORBMethod
from TMCoeffNormed import TMCoeffNormedMethod
from TMNormedCCorr import TMNormedCCorrMethod
from TMSquareDiff import TMSquareDiffMethod
from CustomTM import CustomTMMethod

img_folder_name = "dataset_alter/dataset1/noise"
temp_folder_name = "dataset/dataset1_templates/128"
output_folder = "../results/dataset1_noise_128/"
output_name = "output_noise_128"

match_valid = True # True pour rentrer directement la validation du match, False pour le faire manuellement ultérieurement

img_files = {}
img_and_temp_file = {}

def create_folder(path):
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s" % path)
        
create_folder(output_folder)


#On récupère tous les images (modèle) qu'on stocke dans un dictionnaire ...
for file in os.listdir("../storage/" + img_folder_name):
    if file.endswith(".png") or file.endswith(".jpeg"):
        img_and_temp_file[file] = []
        
# ... et on y ajoute les templates associés     
for file in os.listdir("../storage/" + temp_folder_name):
    if file.endswith(".png") or file.endswith(".jpeg"):
        for img_file in img_and_temp_file:
            file_name_only = os.path.basename(img_file)[:-4]
            if(file.startswith(file_name_only + "_")):
                img_and_temp_file[img_file].append(file)
                
#Affiche les images et templates associés             
#print(img_and_temp_file)



feature_based = [ SIFTMethod(), ORBMethod(),
                 TMCoeffNormedMethod(), TMNormedCCorrMethod(), TMSquareDiffMethod(),
                 CustomTMMethod()]
out_csv_dict = {}

for img_path_name in img_and_temp_file:
    for temp_path_name in img_and_temp_file[img_path_name]:
        
        img_path = "../storage/" + img_folder_name +'/'+ img_path_name
        temp_path = "../storage/" + temp_folder_name +'/'+  temp_path_name
        
        #print('Matching files : img -> ' + img_path + ' & template -> ' + temp_path)
        print("NEXT TEMPLATE")
        
        out_csv_dict[img_path + ' ' + temp_path ] = {} 
        
        for f_b in feature_based:
            
            #si le grayscale est activé sur la méthode on lit l'image grisée
            if(f_b.grayscale):
                img_full = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
                img_temp = cv.imread(temp_path,cv.IMREAD_GRAYSCALE)
            else:
                img_full = cv.imread(img_path,0)
                img_temp = cv.imread(temp_path,0)
            
            f_b.set_pictures(img_full, img_temp)
            key_f_b = f_b.name 
            
            out_path = output_folder + img_path_name[:-4] + "_"+temp_path_name[:-4] +"_"+ key_f_b +".png"
            execution_time = f_b.match(out_path)
            
            out_csv_dict[img_path + ' ' + temp_path ][key_f_b] = {}
            #out_csv_dict[img_path + ' ' + temp_path ][key_f_b]['nb_matches']  = matches
            out_csv_dict[img_path + ' ' + temp_path ][key_f_b]['execution_time']  = execution_time
            
            if(match_valid) :
                value = input("Valid ? (0/1):\n")
 
                valid = value.startswith("1") 
                print(f'You entered {valid}')
                out_csv_dict[img_path + ' ' + temp_path ][key_f_b]['valid_match']  = valid
            else:
                out_csv_dict[img_path + ' ' + temp_path ][key_f_b]['valid_match']  = None

                    
               
    

# on écrit les résultats
with open('../results/'+output_name+'.csv', 'w+', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    filewriter.writerow([
                "Image number",
                "Picture",
                "Method",
                #"Match count",
                "Execution time (s)",
                "Valid match"
            ])
    itera = 0
    for key in out_csv_dict: # filenames
        itera+=1
        for key2 in out_csv_dict[key]: # methods (SURF, SIFT,..)
            filewriter.writerow([
                str(itera),
                str(key),
                str(key2),
                #str(out_csv_dict[key][key2]['nb_matches']),
                str(out_csv_dict[key][key2]['execution_time']),
                str(out_csv_dict[key][key2]['valid_match'])
            ])



#cv.imshow("NORMAL",image)

#cv.waitKey(0)
#cv.destroyAllWindows()
