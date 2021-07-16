# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:39:02 2021

@author: abc
"""

"""


Histogram Equalization and CLAHE ( Contrast Limited Adaptive Histogram Equalization ) to improve contrast in images



"""

#HISTOGRAM EQUALIZATION

import cv2
from skimage import io
from matplotlib import pyplot as plt

#read our images
img = cv2.imread("bio_low_contrast.jpg", 1)
#img = cv2.imread("retinal image proc.jpg", 1)

#Convert our image into Lab space
lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

#Split the chennels of LAB
l, a, b = cv2.split(lab_img)

#plot histogram
plt.hist(l.flat, bins=100, range=(0,255))

#Apply histogram Equalization and plot it
equ = cv2.equalizeHist(l)
plt.hist(equ.flat, bins=100, range=(0,255))

#Now let's visualiize the image
plt.imshow(equ, cmap="gray")

#Let's merge equalize histogram channel (l) with a and b
updated_lab_img1 = cv2.merge((equ, a ,b))

#convert our image into LAB to BGR so we get our original image
hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)


#######################################################################


## CLAHE ##

#Apply CLAHE to L channel
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_img = clahe.apply(l)
#plt.hist(clahe_img.flat, bins=100, range=(0,255))

#Combine the CLAHE enhanced L-channel back with A and B channels
updated_lab_img2 = cv2.merge((clahe_img, a, b))

#Convert LAB image back to color(RGB)
CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)

#Let's see all the images
cv2.imshow("Original image", img)
cv2.imshow("Equalized image", hist_eq_img)
cv2.imshow("CLAHE image", CLAHE_img)
cv2.waitKey(0)
cv2.destroyAllWindows()







