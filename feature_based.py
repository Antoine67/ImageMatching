# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:20:54 2021

@author: Antoine
"""
import cv2 as cv

image = cv.imread("graf1.png")
gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d_SIFT.create()
surf = cv.xfeatures2d_SURF.create()
orb = orb = cv.ORB_create()


keyPoints = sift.detect(image,None)

output = cv.drawKeypoints(image,keyPoints,None)

cv.imshow("FEATURES DETECTED",output)
cv.imshow("NORMAL",image)

cv.waitKey(0)
cv.destroyAllWindows()
