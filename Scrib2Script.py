# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:05:32 2019

@author: Sanket Dhabale
"""

import numpy as np
import cv2
import pytesseract
import imutils
import random


#Read the RAW-IMAGE
img= cv2.imread("F:\PROJECTS\shape-detection\datasetpng\img5.png",1)

cv2.imshow('Original Pic', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(T, thresh) = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Threshold", thresh)

_, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)    

thresh2 =cv2.drawContours(thresh,contours,-1,(0,255,0),2)
cv2.imshow("Threshold with contours", thresh2)

filtered=[]

for c in contours:
	if cv2.contourArea(c) > 10000:filtered.append(c)
	

#Initialize an equally shaped image
objects = np.zeros([img.shape[0],img.shape[1],3], 'uint8')


# i is number of contours in an image
contour_number_i=0

for c in filtered:
    rect=cv2.minAreaRect(c)
    box=cv2.boxPoints(rect)
    box=np.int0(box)
    col = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
    img= cv2.drawContours(objects,[c], -1, col, -1)
    x,y,w,h = cv2.boundingRect(c)
    crop = img[y:(y+h),x:(x+w)]
    cv2.imshow('Features', img)
    cv2.imwrite(str(contour_number_i)+'.png', img)
    contour_number_i+=1