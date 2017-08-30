import numpy as np
import cv2
import convolution
import gaussian_pyramid as gp

class laplacian_pyramid:
    pyramid = []
    def __init__(self, img, levels, kernel = None, gauss_kernel_par = 0.3):
        self.img = img
        self.levels = levels
        if kernel == None:
            self.kernel = convolution.gaussian_kernel(gauss_kernel_par)
        else:
            self.kernel = kernel    
            
    def gauss_operation(self, gauss_cur, gauss_down, operation = '-'):
        if len(gauss_cur.shape) == 3:
            for depth in range(gauss_cur.shape[2]):
                gauss_cur[:, :, depth] = self.gauss_operation(gauss_cur[:, :, depth], gauss_down[:, :, depth], operation)
            return gauss_cur
        if gauss_cur.shape[0] != gauss_down.shape[0]:
                gauss_down = np.vstack((gauss_down, np.full((1, gauss_down.shape[1]), 255)))
        if gauss_cur.shape[1] != gauss_down.shape[1]:
                gauss_down = np.hstack((gauss_down, np.full((gauss_down.shape[0], 1), 255)))
        if operation == '-':
            return cv2.subtract(gauss_cur, gauss_down)
        else:
			return cv2.add(gauss_cur, gauss_down)

    def up_sample(self, image, size):
        gaussian_pyramid = gp.gaussian_pyramid(image, 1)
        image_upsampled = np.empty(size)
        if len(size) == 3:
            for depth in range(size[2]):
                image_upsampled[:,:,depth] = gaussian_pyramid.up_sample(image[:,:,depth], size)
        else:
            image_upsampled = gaussian_pyramid.up_sample(image, size)
        return image_upsampled

    def down(self, up_level_img, cur_level_img):
        if len(up_level_img) == 3:
            new_size= (up_level_img.shape[0] * 2, up_level_img.shape[1] * 2, up_level_img.shape[2])
        else:
            new_size= (up_level_img.shape[0] * 2, up_level_img.shape[1] * 2)
        return self.gauss_operation(cur_level_img.astype(int), self.up_sample(up_level_img, new_size).astype(int), '+')

    def up(self, level, gaussian_pyramid):
        level += 1
        gauss_cur = gaussian_pyramid.get(level)
        gauss_down = gaussian_pyramid.down(level + 1)
        return self.gauss_operation(gauss_cur.astype(int), gauss_down.astype(int), operation='-')
    
    def build(self):
        self.pyramid = []
        gaussian_pyramid = gp.gaussian_pyramid(self.img, self.levels + 1, kernel = self.kernel)
        gaussian_pyramid.build()
        for i in range(-1, self.levels -1):
            self.pyramid.append(self.up(i, gaussian_pyramid))
        #set last level of laplace pyramid = last level of gauss pyramid
        self.pyramid[self.levels - 1] = gaussian_pyramid.get(self.levels - 1)
    
    def reconstruct(self):
        reconstructed = self.get(self.levels - 1)
        for i in range(self.levels - 2, -1, -1):
            reconstructed = self.down(reconstructed, self.get(i))
        return reconstructed

    def get(self, level):
        if(level >=0 and level < self.levels):
            return self.pyramid[level]
        exit('Parameter error: invalid laplace pyramid level')
        
    def show(self, name = 'laplace_pyramid'):
        for i in range(self.levels):
            print(self.get(i).shape)
            cv2.imwrite(name + str(i) + '.jpg', self.get(i))
    

if __name__ == "__main__":
    image_names = ['input/p1-1-0.jpg', 'input/p1-1-1.jpg', 'input/p1-1-2.png', 'input/p1-1-3.png']
    name_it = 0
    pyramid_levels = 7
    for image_name in image_names:
        image = cv2.imread(image_name, 1)
        pyramid = laplacian_pyramid(image, levels = pyramid_levels)
        pyramid.build() 
        #plot downsample results
        print('laplacian pyramid')
        for i in range(pyramid_levels):
            cv2.imwrite('output/p1-2-3-' + str(name_it) + '.jpg', pyramid.get(i))
            print(image_name, 'output/p1-2-3-' + str(name_it) + '.jpg')
            name_it += 1

        #plt reconstruction using laplace pyramid
        cv2.imwrite('output/p1-2-3-' + str(name_it) + '.jpg', pyramid.reconstruct())
        print(image_name, 'reconstruction', 'output/p1-2-3-' + str(name_it) + '.jpg')
        name_it += 1
