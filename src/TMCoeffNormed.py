from generic_method import GenericMethod
import cv2 as cv
import matplotlib.pyplot as plt
import time



class TMCoeffNormedMethod(GenericMethod):
    
    tm_method = cv.TM_CCOEFF
    
    def __init__(self):
        self.name= "TM_CCOEFF" 
        self.sift = cv.xfeatures2d_SIFT.create()
        
        
    def match(self):
        start_time = time.time()
        img1 = self.img_temp
        img2 = self.img_full
        
        res = cv.matchTemplate(img2,img1,self.tm_method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        
        #TODO Add threshold
        
        # Draw first 10 matches.
        #img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #plt.imshow(img3),
        #plt.show()
        
        #return nb_matches, execution_time
        return len(matches), time.time() - start_time