import cv2
import numpy as np

def question_A(img):
	sigma = 0.3
	mu = img.sum()/(1. * img.shape[0] * img.shape[1])
	img[:, :, 1] = img[:, :, 1] + np.random.normal(mu, sigma, (img.shape[0], img.shape[1]))
	cv2.imwrite('../output/p0-5-a-0.jpg', img)

def question_B(img):
  sigma = 0.3
  mu = img.sum()/(1. * img.shape[0] * img.shape[1])
  img[:, :, 0] = img[:, :, 0] + np.random.normal(mu, sigma, (img.shape[0], img.shape[1]))
  cv2.imwrite('../output/p0-5-b-0.jpg', img)

if __name__=='__main__':
	#read image
 	img = cv2.imread('../input/p0-1-3.jpg')
	#solve question
	print img.shape
	question_A(img.copy())
	question_B(img.copy())

