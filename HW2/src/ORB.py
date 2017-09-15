import cv2
import numpy as np
import fast
import matplotlib.pyplot as plt
import math
from scipy.signal import gaussian

'''
  Compute the response(score) of the detector at each pixel(interest points)
'''
def harris_score(sobelx, sobely, row, col, gaussian_kernel, N=5):
  M = [[0,0],[0,0]]
  row = (int)(row - (N-1)/2)
  col = (int)(col - (N-1)/2)
  for i in range(0,N):
    new_row = (int)(row + i)
    for j in range(0,N):
        new_col = (int)(col + j)
        if new_row < 0 or new_row >= sobelx.shape[0] or new_col < 0 or new_col >= sobelx.shape[1]:
            return -1
        if row!= new_row or col!=new_col:
            M[0][0] = M[0][0] + gaussian_kernel[i,j] * (sobelx[new_row,new_col] * sobelx[new_row,new_col])
            M[0][1] = M[0][1] + gaussian_kernel[i,j] * (sobelx[new_row,new_col] * sobely[new_row,new_col])
            M[1][0] = M[1][0] + gaussian_kernel[i,j] * (sobelx[new_row,new_col] * sobely[new_row,new_col])
            M[1][1] = M[1][1] + gaussian_kernel[i,j] * (sobely[new_row,new_col] * sobely[new_row,new_col])

  detM = (M[0][0]*M[1][1]) - (M[1][0]*M[0][1])
  traceM = M[0][0]+M[1][1]
  k = 0.04
  R = detM - k*(traceM*traceM)
  return R

'''
  Harris corner measure, because FAST does not produce a measure of cornerness
'''
def harris_measure_and_orientation(image, interest_points, N):
  score,new_interest_points=[],[]
  gaussian_array0 = gaussian((int)(5), std = 0.5)
  gaussian_array1=[]
  for index in range(len(gaussian_array0)):
    gaussian_array1.append([gaussian_array0[index]])
  gaussian_kernel = np.multiply(gaussian_array0, gaussian_array1)

  gaussian_kernel /= gaussian_kernel.sum()
  sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
  sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3) 
  for index in range(0,len(interest_points)):
     row = int(interest_points[index].pt[1])
     col = int(interest_points[index].pt[0])
     score_ = harris_score(sobelx, sobely, row, col, gaussian_kernel)
     if score_ != -1:
        score.append((score_,index))  
        
  score.sort()
  for index in range(0,N):
    if index>=len(score):
      break
    new_index=score[index][1]
    new_interest_points.append(interest_points[new_index])
    new_interest_points[index].angle = orientation(image, new_interest_points[index], gaussian_kernel, 5)

  return new_interest_points    

'''
  Orientation by Intensity Centroid
'''
def orientation(image, interest_points, gaussian_kernel, size=5):
  hist = np.zeros((36))
  row = int(interest_points.pt[1]) - (size-1)//2
  col = int(interest_points.pt[0]) - (size-1)//2
  for i in range(0,size):
    new_row = row + i
    flag = True
    for j in range(0,size):
        new_col = col + j
        if new_row < 0 or new_row >= image.shape[0] or new_col < 0 or new_col >= image.shape[1]:
            flag = False
        mag = np.sqrt((image[new_row + 1, new_col]*1.0-image[new_row-1,new_col]*1.0)**2 + (image[new_row,new_col+1]*1.0 - image[new_row,new_col-1]*1.0)**2)
        ang = np.arctan2((image[new_row,new_col+1]*1.0-image[new_row, new_col-1]*1.0), (image[new_row + 1, new_col]*1.0-image[new_row-1,new_col]*1.0))
        if ang < 0.0 :
            ang = (ang + 2 * np.pi) 
            ang = (ang*180.0) / np.pi
    if flag:
        hist[(int)(ang/10.0)] += gaussian_kernel[i,j]*mag
  angle = hist.max()

  for index in range(0, 36):
    if angle == hist[index]:
        return (index+1)*10

  return 0

'''
  Build the descriptor
'''
def descriptor(image, interest_points, N=100):
  array_descriptor=[]
  new_interest_points = harris_measure_and_orientation(image, interest_points, N) 
  return 0

if __name__ == "__main__":
  images_names = ['input/fumar.jpg']
  image = cv2.imread(images_names[0], 0)
  keyPoints=fast.interest_points(image, threshold = 30, N = 8)
  keyPoints = harris_measure_and_orientation(image, keyPoints, 400)
  
  #for index in range(0,len(keyPoints)):
   # print(keyPoints[index].pt[0],keyPoints[index].pt[1],keyPoints[index].angle)

  plt.imshow(cv2.drawKeypoints(image, keyPoints, color=(0,255,0), flags=0))
  plt.show()

