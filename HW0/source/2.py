import numpy as np
import cv2

# Load an color image
img = cv2.imread('p0-1-0.jpg')

cv2.imshow('dst_rt', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
