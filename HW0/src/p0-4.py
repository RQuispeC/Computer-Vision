import cv2
import numpy as np

def question_A(img):
	img_max = img.max()
	img_min = img.min()
	img_mean = img.sum()/(1. * img.shape[0] * img.shape[1])
	img_std = np.std(img.ravel())
	print img_max, img_min, img_mean, img_std

def question_B(img):
	img_mean = img.sum()/(1. * img.shape[0] * img.shape[1])
        img_std = np.std(img.ravel())
	img_normalized = ((img - np.full((img.shape[0], img.shape[1]), img_mean))/img_std)*10.0 + np.full((img.shape[0], img.shape[1]), img_mean)
	cv2.imwrite('../output/p0-4-b-0.jpg', img_normalized)

def question_C(img):
	img_shifted = np.hstack((img[:, 2:img.shape[1]], np.full((img.shape[0], 2), 0)))
	img_diff = img - img_shifted
	cv2.imwrite('../output/p0-4-c-0.jpg', img_diff)

if __name__=='__main__':
	#read image
 	img = cv2.imread('../output/p0-2-b-0.jpg', False)
	#solve question
	question_A(img)
	question_B(img)
 	question_C(img)
