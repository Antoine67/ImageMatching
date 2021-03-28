
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:03:34 2021

@author: Nicolas
"""
from generic_method import GenericMethod
import cv2 as cv
import matplotlib.pyplot as plt
import time
import numpy as np

class CustomTMMethod(GenericMethod):
        
    def __init__(self):
        self.name= "CUSTOM_TM_METHOD" 
        
        
    def match(self, output_write_path= None):
        start_time = time.time()
        
        top_left, bottom_right = None, None
        
        zoomList = [ 0.8, 0.9, 1, 1.1, 1.2 ]
        resultArray = []
        THRESHOLD = 0.9
        img = self.img_full
        template = self.img_temp
        
        img_blurred = cv.blur(img,(4,4))
        w, h = template.shape[1], template.shape[0]
        
        for zoom in zoomList:
            
            temp_img = img_blurred.copy()
            temp_template = template.copy()
        
            resize = cv.resize(temp_template, (0, 0), fx=zoom, fy=zoom)
            temp_template = cv.blur(resize,(4,4))
            
            result = cv.matchTemplate(temp_img, temp_template, cv.TM_CCOEFF_NORMED)
            ret,res = cv.threshold(result,THRESHOLD,1,cv.THRESH_TOZERO)
            
            
            _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(res)
            
            if( _maxVal >= 0.9):
                top_left = maxLoc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                
                toSave = img.copy()
                cv.rectangle(toSave, top_left, bottom_right, 255, 2)
                resultArray.append([ _maxVal,toSave])     
                
        bestMatch = None
        for match in resultArray:
            if(bestMatch == None or match[0] > bestMatch[0]):
                bestMatch = match
            
        if(output_write_path):
            if(bestMatch):
                plt.imshow(bestMatch[1]),
                plt.show()
                cv.imwrite(output_write_path,bestMatch[1])
            else:
                print('no match found for ', output_write_path)
                cv.imwrite(output_write_path, img)
        
        #return nb_matches, execution_time
        return time.time() - start_time, top_left, bottom_right