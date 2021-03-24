# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:20:54 2021

@author: Antoine
"""
import cv2 as cv
import csv
from SIFT import SIFTMethod
from TMCoeffNormed import TMCoeffNormedMethod


files = [["../graf1.png","../word.png"],["../word.png","../graf1.png"]]


s =  SIFTMethod()
tm_coeff_normed = TMCoeffNormedMethod()



feature_based = [s, tm_coeff_normed] #, "SURF":surf,"ORB":orb}
out_csv_dict = {}

for file_list in files:
    
    file = file_list[0]
    file2 = file_list[1]
    
    img_full = cv.imread(file,cv.IMREAD_GRAYSCALE)
    img_temp = cv.imread(file2,cv.IMREAD_GRAYSCALE)
    s.set_pictures(img_full, img_temp)
    out_csv_dict[file] = {} 
    
    
    
    for f_b in feature_based:
        
        key_f_b = f_b.name 
        
        matches, execution_time = f_b.match()
        
        out_csv_dict[file][key_f_b] = {}
        out_csv_dict[file][key_f_b]['nb_matches']  = matches
        out_csv_dict[file][key_f_b]['execution_time']  = execution_time
        
        #cv.imshow("FEATURES KEYPOINTS - " + str(f_b),output)
        
    


with open('../results/output-feature-based.csv', 'w+', newline='') as csvfile:
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

'''

#cv.imshow("NORMAL",image)

#cv.waitKey(0)
#cv.destroyAllWindows()'''
