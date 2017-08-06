import numpy as np
import cv2

def replacements_pixels(image, image_green, image_red,number):
	# File name of output
	file_name_output = 'output/p0-3-a-'+str(number)+'.jpg'
	# Index of the central position	
	fil=int(image_green.shape[0]/2-50)
	col=int(image_green.shape[1]/2-50)
	#	Replace indexs in original image
	image_green[fil:fil+100,col:col+100]=image_red[fil:fil+100,col:col+100]
	#	Write output image
	cv2.imwrite(file_name_output,image_green)
	# replace of channel green in the original image	
	replacements_original(image, image_green, number)

def replacements_original(image, image_green, number):
	# File name of output
	file_name_output = 'output/p0-3-b-'+str(number)+'.jpg'
	#	Replace indexs
	image[:,:,1]=image_green
	#	Write output image
	cv2.imwrite(file_name_output,image)

if __name__ == "__main__":
	for i in range(0,4):
		# File name of inputs
		file_name_input = 'input/p0-1-'+str(i)+'.jpg'
		file_name_input1 = 'output/p0-2-c-'+str(i)+'.jpg'
		file_name_input2 = 'output/p0-2-b-'+str(i)+'.jpg'
		# Load an color original image
		image = cv2.imread(file_name_input)
		# Load grayscale images
		image_red = cv2.imread(file_name_input1,0)
		image_green = cv2.imread(file_name_input2,0)
		# Replace center
		replacements_pixels(image, image_green, image_red , i)

