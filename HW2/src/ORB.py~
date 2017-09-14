import cv2
import numpy
import fast
import matplotlib.pyplot as plt
import math

dx=[-1,-1,0,1,1,1,0,-1]
dy=[0,1,1,1,0,-1,-1,-1]

'''
  Compute the response(score) of the detector at each pixel(interest points)
'''
def harris_score(sobelx, sobely, row, col):
  M = [[0,0],[0,0]]
  flag = False
  for index in range(0,8):
    new_row = row + dx[index]
    new_col = col + dy[index]
    if new_row < 0 or new_row >= sobelx.shape[0] or new_col < 0 or new_col >= sobelx.shape[1]:
      flag = True
      break 
    M[0][0] = M[0][0] + (sobelx[new_row,new_col] * sobelx[new_row,new_col])
    M[0][1] = M[0][1] + (sobelx[new_row,new_col] * sobely[new_row,new_col])
    M[1][0] = M[1][0] + (sobelx[new_row,new_col] * sobely[new_row,new_col])
    M[1][1] = M[1][1] + (sobely[new_row,new_col] * sobely[new_row,new_col])
  
  if flag :
    return 0

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
  sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
  sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3) 
  for index in range(0,len(interest_points)):
     row = int(interest_points[index].pt[1])
     col = int(interest_points[index].pt[0])
     score.append((harris_score(sobelx, sobely, row, col),index))  
  
  score.sort()
  for index in range(0,N):
    if index>=len(score):
      break
    new_index=score[index][1]
    new_interest_points.append(interest_points[new_index])
    new_interest_points[index].angle = orientation(image, new_interest_points[index])
    
  return new_interest_points

'''
  Orientation by Intensity Centroid
'''
def orientation(image, interest_points):
  row = int(interest_points.pt[1])
  col = int(interest_points.pt[0])
  m01, m10 = 0, 0
  for index in range(0, 8) :
    new_row = row + dx[index]
    new_col = col + dy[index]
    m01 = m01 + new_row * image[new_row,new_col]
    m10 = m10 + new_col * image[new_row,new_col]

  return math.atan2(m01,m10)

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
