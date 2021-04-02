# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:20:54 2021

@author: Antoine
"""
import cv2 as cv
import pandas as pd
import csv
import os
from SIFT import SIFTMethod
from ORB import ORBMethod
from TMCoeffNormed import TMCoeffNormedMethod
from TMNormedCCorr import TMNormedCCorrMethod
from TMSquareDiff import TMSquareDiffMethod
from CustomTM import CustomTMMethod

img_folder_name = "dataset_alter/dataset1"
temp_folder_name = "dataset/dataset1_templates"
output_name = "output"
save_picture = False


template_sizes = [256]
altered_types = ['rotation','normal','blur', 'zoom', 'rotation', 'noise']

feature_based = [ SIFTMethod(), ORBMethod(),
                  TMCoeffNormedMethod(), TMNormedCCorrMethod(), TMSquareDiffMethod()]
                  #CustomTMMethod()]

def create_folder(path):
           try:
               os.makedirs(path)
           except OSError:
               print ("Creation of the directory %s failed" % path)
           else:
               print ("Successfully created the directory %s" % path)


ERROR_RATE_PERCENT = 0.05
def isValidMatch(img_path, temp_path, top_left, bottom_right, df_template_creation, method, height, width):
    df = df_template_creation.copy()

    # filtering data
    df = df.loc[df['image'] == img_path]
    df = df.loc[df['template'] == temp_path]
    
    df = df.iloc[0] # on ne garde que la première ligne match (si plusieurs)
    
    #print(top_left,bottom_right, '\n',df)

    x1_key, y1_key, x2_key, y2_key = "x1", "y1", "x2", "y2"
   
    if method == 'zoom' and {'x1_zoom', 'y1_zoom', 'x2_zoom','y2_zoom'}.issubset(df_template_creation.columns):
        x1_key, y1_key, x2_key, y2_key = 'x1_zoom', 'y1_zoom', 'x2_zoom','y2_zoom'
    elif method == 'rotation' and {'x1_rotation', 'y1_rotation', 'x2_rotation','y2_rotation'}.issubset(df_template_creation.columns):
        x1_key, y1_key, x2_key, y2_key = 'x1_rotation', 'y1_rotation', 'x2_rotation','y2_rotation'
    
        
        

    try:
        
        print( top_left[0], df[x1_key] ,
                top_left[1] , df[y1_key],
                bottom_right[0] , df[x2_key], 
                bottom_right[1] , df[y2_key] )
        
        
        '''
        print(abs(top_left[0] - df.x1) , width * ERROR_RATE_PERCENT,
                abs(top_left[1] - df.y1) , width * ERROR_RATE_PERCENT ,
                abs(bottom_right[0] - df.x2) , height * ERROR_RATE_PERCENT ,
                abs(bottom_right[1] - df.y2) , height * ERROR_RATE_PERCENT)
        '''
        return  abs(top_left[0] - df[x1_key]) < width * ERROR_RATE_PERCENT and \
                abs(top_left[1] - df[y1_key]) < width * ERROR_RATE_PERCENT and \
                abs(bottom_right[0] - df[x2_key]) < height * ERROR_RATE_PERCENT and \
                abs(bottom_right[1] - df[y2_key]) < height * ERROR_RATE_PERCENT
        '''
        return  (top_left[0] < df[x1_key]  * ERROR_RATE_PERCENT ) and \
                (top_left[1] < df[y1_key] * ERROR_RATE_PERCENT ) and \
                (bottom_right[0] < df[x2_key] * ERROR_RATE_PERCENT ) and \
                (bottom_right[1] < df[y2_key] * ERROR_RATE_PERCENT ) 
        '''
    except:
        return False


img_and_temp_file = {}
#On récupère tous les images (modèle) qu'on stocke dans un dictionnaire ...
for file in os.listdir(f"../storage/{img_folder_name}/normal"):
    if file.endswith(".png") or file.endswith(".jpeg"):
        img_and_temp_file[file] = []

for t_size in template_sizes:
       
    # on supprime les eventuels templates précèdents
    for value in img_and_temp_file.values():
        del value[:]
        
    # ... et on y ajoute les templates associés     
    for file in os.listdir(f"../storage/{temp_folder_name}/{t_size}"):
        if file.endswith(".png") or file.endswith(".jpeg"):
            for img_file in img_and_temp_file:
                file_name_only = os.path.basename(img_file)[:-4]
                if(file.startswith(file_name_only + "_")):
                    img_and_temp_file[img_file].append(file)
                    
      
    df_template_creation = pd.read_csv(f"../results/template_creation_after_pipeline_{t_size}.csv",header=0, dtype=object,sep=';')
    
    df_template_creation['image'] = df_template_creation['image'].astype(str)
    df_template_creation['template'] = df_template_creation['template'].astype(str)
    
    df_template_creation['x1'] = df_template_creation['x1'].astype(int)
    df_template_creation['y1'] = df_template_creation['y1'].astype(int)
    df_template_creation['x2'] = df_template_creation['x2'].astype(int)
    df_template_creation['y2'] = df_template_creation['y2'].astype(int)
    
    df_template_creation['x1_rotation'] = df_template_creation['x1_rotation'].astype(float)
    df_template_creation['y1_rotation'] = df_template_creation['y1_rotation'].astype(float)
    df_template_creation['x2_rotation'] = df_template_creation['x2_rotation'].astype(float)
    df_template_creation['y2_rotation'] = df_template_creation['y2_rotation'].astype(float)
    
    df_template_creation['x1_zoom'] = df_template_creation['x1_zoom'].astype(float)
    df_template_creation['y1_zoom'] = df_template_creation['y1_zoom'].astype(float)
    df_template_creation['x2_zoom'] = df_template_creation['x2_zoom'].astype(float)
    df_template_creation['y2_zoom'] = df_template_creation['y2_zoom'].astype(float)
       
    
    for a_type in altered_types:
         
        if(save_picture):
            output_folder = f"../results/output_folder/{t_size}/"
            create_folder("../results/output_folder")
            create_folder(output_folder)
       
        out_csv_dict = {}
        
        for img_path_name in img_and_temp_file:
            temp_c = 0
            for temp_path_name in img_and_temp_file[img_path_name]:
                
                img_path = f"../storage/{img_folder_name}/{a_type}/{img_path_name}"
                temp_path = f"../storage/{temp_folder_name}/{t_size}/{temp_path_name}"
                
                print('Matching files : img -> ' + img_path + ' & template -> ' + temp_path)
                        
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
                    
                   
                    out_path = None
                    if(save_picture):
                        out_path = output_folder + img_path_name[:-4] + "_"+temp_path_name[:-4] +"_"+ key_f_b +".png"
                        print('Saving as ', out_path)
                    
                    try:
                        execution_time, top_left, bottom_right = f_b.match(out_path)
                    except:
                        execution_time, top_left, bottom_right = None, None, None
                        
                    valid_match = isValidMatch(img_path_name, temp_path_name,top_left, bottom_right, df_template_creation,a_type, img_full.shape[0], img_full.shape[1])
                    
                    print("Matching :",img_path_name, temp_path_name,top_left, bottom_right, valid_match)
                    
                    out_csv_dict[img_path + ' ' + temp_path ][key_f_b] = {}
                    #out_csv_dict[img_path + ' ' + temp_path ][key_f_b]['nb_matches']  = matches
                    out_csv_dict[img_path + ' ' + temp_path ][key_f_b]['execution_time']  = execution_time
                    
                    out_csv_dict[img_path + ' ' + temp_path ][key_f_b]['valid_match']  =  valid_match
                    
                    
                        
        
        # on écrit les résultats
        with open(f'../results/{output_name}_{a_type}_{t_size}.csv', 'w+', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
            filewriter.writerow([
                        "Image number",
                        "Picture",
                        "Method",
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
