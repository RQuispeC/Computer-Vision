import numpy as np
import cv2
import convolution
import scipy as sp
from scipy.interpolate import interp2d
import gaussian_pyramid as gp

class laplacian_pyramid:
    pyramid = []
    def __init__(self, img, levels, kernel_a = 0.3):
        self.img = img
        self.levels = levels
        self.gaussian_pyramid = gp.gaussian_pyramid(img, levels + 1, kernel_a)
    
    def operation_gauss(self, gauss_cur, gauss_down, operation = '-'):
        if len(gauss_cur.shape) == 3:
            gauss_cur[:, :, 0] = self.operation_gauss(gauss_cur[:, :, 0], gauss_down[:, :, 0], operation)
            gauss_cur[:, :, 1] = self.operation_gauss(gauss_cur[:, :, 1], gauss_down[:, :, 1], operation)
            gauss_cur[:, :, 2] = self.operation_gauss(gauss_cur[:, :, 2], gauss_down[:, :, 2], operation)
            return gauss_cur
        if gauss_cur.shape[0] != gauss_down.shape[0]:
                gauss_down = np.vstack((gauss_down, np.full((1, gauss_down.shape[1]), 255)))
        if gauss_cur.shape[1] != gauss_down.shape[1]:
                gauss_down = np.hstack((gauss_down, np.full((gauss_down.shape[0], 1), 255)))
        if operation=='-':
            return cv2.subtract(gauss_cur,gauss_down)
        else:
			return cv2.add(gauss_cur,gauss_down)

    def down(self, level):
        lapl_cur = self.get(level)
        gauss_up = self.gaussian_pyramid.down(level+1)
        return self.operation_gauss(lapl_cur.astype(int), gauss_up.astype(int), '+')
    
    def up(self, level):
        level += 1
        gauss_cur = self.gaussian_pyramid.get(level)
        gauss_down = self.gaussian_pyramid.down(level + 1)
        return self.operation_gauss(gauss_cur.astype(int), gauss_down.astype(int), operation='-')
    
    def build(self):
        self.pyramid = []
        self.gaussian_pyramid.build()
        for i in range(-1, self.levels -1):
            self.pyramid.append(self.up(i))
    
    def get(self, level):
        if(level >=0 and level < self.levels):
            return self.pyramid[level]
        exit('Parameter error: invalid laplace level')
        
    def show(self, name = 'laplace_pyramid'):
        for i in range(self.levels):
            print(self.get(i).shape)
            cv2.imwrite(name + str(i) + '.jpg', self.get(i))
    

if __name__ == "__main__":
    file_name_input = 'p0-1-0.jpg'
    image = cv2.imread(file_name_input,1) 
    pyramide=laplacian_pyramid(image,6)
    pyramide.build()
    pyramide.show()
    cv2.imwrite('prueba.jpg',pyramide.down(0))
