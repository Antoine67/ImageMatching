import abc
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time


class GenericMethod:

    img_full = None
    img_temp = None
    
    grayscale = False
        
    
    def set_pictures(self, full, template):
        self.img_full = full
        self.img_temp = template
        
        
    @abc.abstractmethod 
    def match(self):
        """Method documentation goes there"""
        return
    
    
    def generic_tm_match(self, output_write_path):
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

        if self.tm_method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        cv.rectangle(img3, top_left, bottom_right, 255, 2)
        

        if(output_write_path):
            plt.imshow(img3),
            plt.show()
            cv.imwrite(output_write_path,img3)
            
             
        
        #return execution_time
        return time.time() - start_time, top_left, bottom_right
    
    def get_rect_feature_based(self,matches, kp1,kp2, img1, img2 ):
        ratio = 0.75
        MIN_MATCH = 7
        good_matches = [m[0] for m in matches \
                            if len(m) == 2 and m[0].distance < m[1].distance * ratio]
        print('good matches:%d/%d' %(len(good_matches),len(matches)))
        
        matchesMask = np.zeros(len(good_matches)).tolist()
        res_points = [None] * 4
        
        
        if len(good_matches) > MIN_MATCH: 
            
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
            
            mtrx, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            accuracy=float(mask.sum()) / mask.size
            print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))
            if mask.sum() > MIN_MATCH:  
                
                matchesMask = mask.ravel().tolist()
               
                h,w, = img1.shape[:2]
                pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
                dst = cv.perspectiveTransform(pts,mtrx)
                res_points = [np.int32(dst)]
                img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
                 
        res = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, \
                            matchesMask=matchesMask,
                            flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
         
        try:
            return res_points[0][0][0], \
                   res_points[0][2][0], \
                   res
        except:
            return None, None, res
        