import numpy as np 
import cv2 
#reading the input image 
img = cv2.imread('./visual/image_29_map.jpg')
img[img<30]=0
img[img>30]=255
closing = img
kernel2 = np.ones((2,2), dtype = "uint8")
kernel1 = np.ones((2,2), dtype = "uint8")

closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2) 
closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel1) 

cv2.imwrite('./visual/image_29_processed.jpg', closing)