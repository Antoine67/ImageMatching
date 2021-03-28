from generic_method import GenericMethod
import cv2 as cv
import time
import matplotlib.pyplot as plt



class SIFTMethod(GenericMethod):
    
    grayscale = True
    
    def __init__(self):
        self.name= "SIFT" 
        self.sift = cv.xfeatures2d_SIFT.create()
        
        
    def match(self, output_write_path= None):
        start_time = time.time()
        img1 = self.img_temp
        img2 = self.img_full
        
        
        kp1, des1 = self.sift.detectAndCompute(img1,None)
        kp2, des2 = self.sift.detectAndCompute(img2,None)
        
        
        FLANN_INDEX_LSH = 5
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                           table_number = 6,
                           key_size = 1,
                           multi_probe_level = 1)
        search_params=dict(checks=32)
        #matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
        
        matches = matcher.knnMatch(des1, des2, 2)
        
        top_left, bottom_right, img_output = self.get_rect_feature_based(matches, kp1,kp2, img1.copy(), img2.copy())
        if(output_write_path):
            plt.imshow(img_output)
            plt.show()
            cv.imwrite(output_write_path,img_output)
                    
        #return execution_time
        return time.time() - start_time, top_left, bottom_right