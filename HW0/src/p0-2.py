import numpy as np
import cv2

def Swap_channels(image_,number):
	file_name_output = 'output/p0-2-a-'+str(number)+'.jpg'
	# Swap blue channel with red channel
	image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
	# Write image in the output folder
	cv2.imwrite(file_name_output,image_)

def Monochrome_Image(image, number):
	for i in range(0,2):
		file_name_output = 'output/p0-2-'+chr(98+i)+'-'+str(number)+'.jpg'
		#	Monochrome image(green and red)
		monochrome = image[:,:,i+1]
		# Write image in the output folder
		cv2.imwrite(file_name_output,monochrome)

if __name__ == "__main__":
	for i in range(0,4):
		# Load an color image
		file_name_input='input/p0-1-'+str(i)+'.jpg'
		image = cv2.imread(file_name_input)
		# Question 1.a
		Swap_channels(image,i)
		# Question 1.b and 1.c
		Monochrome_Image(image,i)
