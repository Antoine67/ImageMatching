from generic_method import GenericMethod
import cv2 as cv
import matplotlib.pyplot as plt
import time
import numpy as np



class TMCoeffNormedMethod(GenericMethod):
    
    tm_method = cv.TM_CCOEFF_NORMED
    
    def __init__(self):
        self.name= "TM_CCOEFF_NORMED" 
        self.sift = cv.xfeatures2d_SIFT.create()
        
        
    def match(self):
        start_time = time.time()
        img1 = self.img_temp
        img2 = self.img_full
        
        w, h = img1.shape[::-1]
        
        res = cv.matchTemplate(img2,img1,self.tm_method)
        #min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        
        threshold = 0.8
        loc = np.where( res >= threshold)
        
        f = set()
        
        for pt in zip(*loc[::-1]):        
            sensitivity = 100
            f.add((round(pt[0]/sensitivity), round(pt[1]/sensitivity)))
        
        
        
        #return nb_matches, execution_time
        return len(f), time.time() - start_time