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
        
        f = set()
        
        for pt in zip(*loc[::-1]):        
            sensitivity = 100
            f.add((round(pt[0]/sensitivity), round(pt[1]/sensitivity)))
            if(output_write_path):
                cv.rectangle(img3, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)  
            
        
        # Draw first 10 matches.
        if(output_write_path):
            plt.imshow(img3),
            plt.show()
            cv.imwrite(output_write_path,img3)
            
            
        
        
        
        #return nb_matches, execution_time
        return len(f), time.time() - start_time