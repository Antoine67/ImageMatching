from generic_method import GenericMethod
import cv2 as cv
import matplotlib.pyplot as plt
import time
import numpy as np



class TMSquareDiffMethod(GenericMethod):
    
    tm_method = cv.TM_SQDIFF
    
    def __init__(self):
        self.name= "TM_SQDIFF" 
        self.sift = cv.xfeatures2d_SIFT.create()
        
        
    def match(self, output_write_path= None):
        start_time = time.time()
        img1 = self.img_temp
        img2 = self.img_full
        img3 = img2.copy() # on copie l'image originale 
        
        w, h = img1.shape[::-1]
        
        res = cv.matchTemplate(img2,img1,self.tm_method)
        #min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        
        threshold = 0.8
        loc = np.where( res >= threshold)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        cv.rectangle(img3, top_left, bottom_right, 255, 2)
        

        if(output_write_path):
            plt.imshow(img3),
            plt.show()
            cv.imwrite(output_write_path,img3)
            
            
        
        
        
        #return execution_time
        return  time.time() - start_time