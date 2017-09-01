import cv2
import numpy as np

img = cv2.imread('../input/p2-1-0.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

print(gray.shape, img.shape)

sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)

cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

print(des)

cv2.imwrite('sift_keypoints.jpg', img)

