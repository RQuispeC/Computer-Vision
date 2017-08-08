import cv2
import numpy as np

def question_A(img):
	img_max = img.max()
	img_min = img.min()
	img_mean = img.sum()/(1. * img.shape[0] * img.shape[1])
	img_std = np.std(img.ravel())
	print ('max:', img_max, 'min:', img_min, 'mean:', img_mean, 'std:', img_std)

def question_B(img, img_cnt):
	img_mean = img.sum()/(1. * img.shape[0] * img.shape[1])
	mean_img = np.full((img.shape[0], img.shape[1]), img_mean)
	img_std = np.std(img.ravel())
	img_normalized = ((img - mean_img)/img_std)*10.0 + mean_img
	cv2.imwrite('output/p0-4-b-'+ str(img_cnt) + '.jpg', img_normalized)

def question_C(img, img_cnt):
	img_shifted = np.hstack((img[:, 2:img.shape[1]], np.full((img.shape[0], 2), 0)))
	cv2.imwrite('output/p0-4-c-'+ str(img_cnt*2) +'.jpg', img_shifted)
	img_diff = img - img_shifted
	cv2.imwrite('output/p0-4-c-'+ str(img_cnt*2+1) +'.jpg', img_diff)

if __name__=='__main__':
	for img_cnt in range(4):
		img = cv2.imread('output/p0-2-b-' + str(img_cnt) + '.jpg', False)
		print ('p0-2-b-' + str(img_cnt) + '.jpg')
		#solve question
		question_A(img)
		question_B(img, img_cnt)
		question_C(img, img_cnt)
