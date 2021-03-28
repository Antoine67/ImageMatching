from generic_method import GenericMethod
import cv2 as cv
import matplotlib.pyplot as plt
import time
import numpy as np



class TMSquareDiffMethod(GenericMethod):
    
    tm_method = cv.TM_SQDIFF
    
    def __init__(self):
        self.name= "TM_SQDIFF" 
        
        
    def match(self, output_write_path= None):
        return self.generic_tm_match(output_write_path)