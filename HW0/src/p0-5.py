import cv2
import numpy as np

def question_A(img, img_cnt):
	sigma = 20
	#mu = img[:, :, 1].sum()/(1. * img.shape[0] * img.shape[1]) * 0.5
	mu = 1
	img[:, :, 1] = img[:, :, 1] + np.random.normal(mu, sigma, (img.shape[0], img.shape[1]))
	cv2.imwrite('output/p0-5-a-'+ str(img_cnt) + '.jpg', img)

def question_B(img, img_cnt):
	sigma = 20
	mu = 1
	img[:, :, 0] = img[:, :, 0] + np.random.normal(mu, sigma, (img.shape[0], img.shape[1]))
	cv2.imwrite('output/p0-5-b-'+ str(img_cnt) +'.jpg', img)

if __name__=='__main__':
	for img_cnt in range(4):
		img = cv2.imread('input/p0-1-' + str(img_cnt) + '.jpg')
		#solve question
		question_A(img.copy(), img_cnt)
		question_B(img.copy(), img_cnt)
