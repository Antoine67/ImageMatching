from generic_method import GenericMethod
import cv2 as cv
import matplotlib.pyplot as plt
import time



class SIFTMethod(GenericMethod):
    
    grayscale = True
    
    def __init__(self):
        self.name= "SIFT" 
        self.sift = cv.xfeatures2d_SIFT.create()
        
        
    def match(self, output_write_path= None):
        start_time = time.time()
        img1 = self.img_temp
        img2 = self.img_full
        
        '''
        gray_temp = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
        gray_full = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)'''
        
        kp1, des1 = self.sift.detectAndCompute(img1,None)
        kp2, des2 = self.sift.detectAndCompute(img2,None)
        
       
        # create BFMatcher object
        bf = cv.BFMatcher()
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        #TODO Add threshold
        
        # Draw first 10 matches.
        if(output_write_path):
            img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img3),
            plt.show()
            cv.imwrite(output_write_path,img3)
        
        #return nb_matches, execution_time
        return len(matches), time.time() - start_time