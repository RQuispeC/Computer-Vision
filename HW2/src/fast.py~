import numpy as np
import random
import cv2
import pandas as pd
import matplotlib.pyplot as plt

dx=[-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3]
dy=[0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1]

def training_set(images_names, threshold = 10, N = 12):
  IP=[]
  for index in range(0, len(images_names)) :
		image = cv2.imread(images_names[index], 0)
		P = interest_points(image, threshold, N)
		IP = IP + P
	
  df = pd.DataFrame(P)
  df.to_csv('RF.csv', index=False,header=False)
  return subset(interest_point)

def subset(interest_point):
	P,P_d,P_s,P_b=[],[],[],[]
	# shuffle x
	shuffle = random.randrange(16)
	for index in range(0, len(interest_point)):
		if interest_point[index][shuffle] == 'd':
			P_d.append(interest_point[index])
		if interest_point[index][shuffle] == 's':
			P_s.append(interest_point[index])
		if interest_point[index][shuffle] == 'b':
			P_b.append(interest_point[index])

	P.extend((P_d,P_s,P_b))

	return P

'''
	FAST: estract the interest points with Features from Accelerated Segment Test
'''
def interest_points(image, threshold = 10, N = 12):
  keyPoints=[]
  for row in range(3,image.shape[0]-3,2) :
    for col in range(3, image.shape[1]-3,2) :
      flag = is_interest_point(image, row, col, threshold, N)
      if flag:
				keyPoints.append((col,row))

  suppressionPoints = non_maximal_suppression(image,keyPoints,threshold)
  print ('=====> {0:2d} Interest Point without non-Maximal Supression'.format(len(keyPoints)))
  print ('=====> {0:2d} Interest Point with non-Maximal Supression'.format(len(suppressionPoints)))

  return suppressionPoints

'''
	Non-Maximal Suppression over interest points
'''
def non_maximal_suppression(image, keyPoints, threshold) :
  cont=0
  keyPointsNMX=[]
  score=np.zeros(image.shape)

  for i in range(0,len(keyPoints)):
		scoreDark, scoreBrig = 0,0
		intensity = image[keyPoints[i][1],keyPoints[i][0]]
		for index in range(0, len(dx)):
			new_row = keyPoints[i][1]+dx[index]
			new_col = keyPoints[i][0]+dy[index]
			if (new_row>=0 and new_row<score.shape[0] and new_col>=0 and new_col<score.shape[1]):
				new_intensity=image[new_row,new_col]
				state = state_pixel(intensity, new_intensity, threshold)
				difference = abs(int(intensity)-int(new_intensity))
				if state == 'd':
					scoreDark = scoreDark + difference
				else :
					if state == 'b':
						scoreBrig = scoreBrig + difference
		score[keyPoints[i][1],keyPoints[i][0]] = max(scoreDark,scoreBrig)

  for i in range(0,len(keyPoints)):
		row = keyPoints[i][1]
		col = keyPoints[i][0]
		maximoScore=score[row,col]
		for k in range(0,16):
			new_row = row + dx[k]
			new_col = col + dy[k]
			if (new_row>=0 and new_row<score.shape[0] and new_col>=0 and new_col<score.shape[1]) and score[new_row,new_col]>maximoScore : 
				maximoScore = score[new_row,new_col]
				break

		if maximoScore == score[row,col] :
			keyPointsNMX.append(cv2.KeyPoint(col,row,5))

  return keyPointsNMX

'''
	Detect if a pixel is or is not a interest point
'''
def is_interest_point(image,row, col, threshold, N) :
	countDark,countBrig=0,0

	for index in range(0,16,4) :
		intensity = image[row,col]
		new_intensity=image[row+dx[index],col+dy[index]]		
		if new_intensity <= intensity-threshold :
			countDark += 1
		if intensity + threshold <= new_intensity:
			countBrig += 1
  # fast analysis with four point
	if countBrig < N/4 and countDark < N/4:
		return False

	countDark,countBrig,countInit=0,0,0
	flagFirst = True
	typePixel=  state_pixel(image[row,col], image[row+dx[0],col+dy[0]], threshold)
	flagIP = False	
	for index in range(0, len(dx)):
		intensity = image[row,col]
		new_intensity=image[row+dx[index],col+dy[index]]
		state = state_pixel(intensity, new_intensity, threshold)
		if flagFirst and state == typePixel:
			countInit = countInit + 1
		else:
			flagFirst = False

		if state == 'd':
			countDark = countDark + 1
			countBrig = 0
		else :
			if state == 'b':
				countBrig = countBrig + 1
				countDark = 0
			else:	
				countBrig = 0
				countDark = 0
		if countBrig >= N or countDark >=N :
			return True

		if (countDark==1 or countBrig==1) and (state!=typePixel) and (16-index<N):
			return False

	if typePixel == state_pixel(image[row,col],image[row+dx[15],col+dy[15]],threshold):
		if typePixel == 'd' and countInit+countDark>=N:
			return True
		else :
			if typePixel == 'b' and countInit+countBrig>=N:
				return True
	return False

'''
	State of a pixel(darker, similar, brighter)
'''
def state_pixel(intensity, new_intensity, threshold):
  if new_intensity <= intensity-threshold :
		return 'd' # darker
  else :
		if intensity - threshold < new_intensity and new_intensity < intensity + threshold :
			return 's' # similar
		else :
			if intensity + threshold <= new_intensity :
				return 'b' # brighter
  return 's'

if __name__ == "__main__":
	images_names = ['input/fumar.jpg']
	image = cv2.imread(images_names[0], 0)
	keyPoints=interest_points(image, threshold = 30, N = 8)
	plt.imshow(cv2.drawKeypoints(image, keyPoints, color=(0,255,0), flags=0))
	plt.show()
