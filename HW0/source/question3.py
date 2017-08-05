import numpy as np
import cv2

def replacements_pixels(image):
	print(image.shape)
if __name__ == "__main__":
	for i in range(0,1):
		# Load an color image
		file_name_input='../output/p0-2-a'+str(number)+'.jpg'
		file_name_input2='../output/p0-2-a'+str(number)+'.jpg'
		image = cv2.imread(file_name_input)
		cv2.imshow('dst_rt', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
