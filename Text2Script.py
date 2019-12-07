# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:31:18 2019

@author: Sanket Dhabale
"""


import numpy as np
import cv2
import pytesseract
import imutils
import random
import pandas as pd


#Read the RAW-IMAGE
img= cv2.imread("F:\PROJECTS\shape-detection\datasetpng\img7.png",1)

#Display RAW-IMAGE in a new window
#cv2.imshow('Original Pic', img)

#Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#applying threshold to remove noise from the grayscale image
(T, thresh) = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#cv2.imshow("Threshold", thresh)



text = pytesseract.image_to_string(thresh)
print(text) 
text = text.replace('\n', ' ').replace('\r', '')

#for writing in the text file
f= open('Attribute.txt','w')
f.write(text)

#for reading the text file
f = open('Attribute.txt','r')
message = f.read()
print(message)


df = pd.read_fwf('Attribute.txt')
df.to_csv('Attribute.csv')

"""


#Finding contours with simple retrieval (no hierarchy) and simple/compressed end points
_, contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)    
    

filtered=[]

#Looping over all found contours
for c in contours:
    #If it has significant area, add to list
   	if cv2.contourArea(c) > 10000:filtered.append(c)
   

#Initialize an equally shaped image
#objects = np.zeros([img.shape[0],img.shape[1],3], 'uint8')


# i is number of contours in an image it belongs to.
#contourlen=[2,3,5,6]
contour_number_i=0 
filtered_copy= filtered
rect_attr=[]
oval_attr=[]

for c in filtered_copy:
     peri = cv2.arcLength(c, True)
     approx = cv2.approxPolyDP(c, 0.04 * peri, True)
     if len(approx) ==4:
        rect_attr.append(c)
     else: oval_attr.append(c)
         #if len(approx) not in contourlen:
         #   oval_attr.append(c)

print(len(filtered_copy),len(oval_attr),len(rect_attr))

for c in rect_attr:
    x,y,w,h = cv2.boundingRect(c)
    crop_table = img[y:(y+h),x:(x+w)]
    gray1 = cv2.cvtColor(crop_table, cv2.COLOR_BGR2GRAY)
    (T, thresh1) = cv2.threshold(gray1, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    table_name_text= pytesseract.image_to_string(thresh1)
    print(table_name_text)
    cv2.imshow("TableName",crop_table)
    print(len(rect_attr))

for c in oval_attr:
    x,y,w,h = cv2.boundingRect(c)
    crop_attr = img[y:(y+h),x:(x+w)]
    gray2 = cv2.cvtColor(crop_attr, cv2.COLOR_BGR2GRAY)
    (T, thresh1) = cv2.threshold(gray2, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    attr_name_text= pytesseract.image_to_string(thresh1)
    print(attr_name_text)
    cv2.imshow("Feature",crop_attr)
    cv2.imwrite(str(contour_number_i)+'.png',crop_attr)
    contour_number_i+=1
    cv2.imshow("AttributeName",crop_attr)
    print(len(oval_attr))
    

#Closing protocol
cv2.waitKey(0)
cv2.destroyAllWindows()




"""