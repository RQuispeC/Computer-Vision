import numpy as np
import cv2
import convolution
import scipy as sp

class gaussian_pyramid:
    pyramid = []              
    def __init__(self, img, levels, kernel = None, gauss_kernel_par= 0.3):
        self.img = img
        self.levels = levels
        if kernel == None:
            self.kernel = convolution.gaussian_kernel(gauss_kernel_par)
        else:
            self.kernel = kernel    
    
    def interpolation(self,x,y,v, interp = 'bilinear'):
        if interp=='bilinear':  
            if (x[0]==x[1]==x[2]==x[3]) or (y[0]==y[1]==y[2]==y[3]):
                return v[0]
            r1=(((x[1]-x[4])/(x[1]-x[0]))*v[2]+(((x[4]-x[0])/(x[1]-x[0])))*v[3])
            r2=(((x[1]-x[4])/(x[1]-x[0]))*v[0]+(((x[4]-x[0])/(x[1]-x[0])))*v[1])
            p=(((y[1]-y[4])/(y[0]-y[2]))*r1+(((y[4]-y[2])/(y[0]-y[2])))*r2)
            return p
        return 1

    def up_sample(self, image, size, interp = 'bilinear'):
        new_image=np.empty(size)
        if len(image.shape) == 3:
            new_image = np.empty((size[0], size[1], image.shape[2]))
            for depth in range(image.shape[2]):
                new_image[:, :, depth] = self.up_sample(image[:, :, depth], (size[0], size[1]), interp ) 
            return new_image
        for i in range(size[0]):
            for j in range(size[1]):
                if(interp == 'bilinear'):
                    i_min = min(i//2, image.shape[0]-1)
                    j_min = min(j//2, image.shape[1]-1)
                    i_lim = min(i_min +1, image.shape[0]-1)
                    j_lim = min(j_min +1, image.shape[1]-1)
                    y = [i_min, i_min, i_lim, i_lim, (i_min + i_lim)/2.0]
                    x = [j_min, j_lim, j_min, j_lim, (j_min + j_lim)/2.0]
                    z = [image[i_min, j_min], image[i_min, j_lim], image[i_lim, j_min], image[i_lim, j_lim]]
                    new_image[i, j] = self.interpolation(x, y, z, interp)
        return new_image
    
    def down_sample(self, image, size, padding_type = 'zero', padding_color = 0):
        new_image=np.empty(size)
        filtered_image=convolution.convolve(image, self.kernel, normalize=False, padding_type = padding_type, padding_color = padding_color)
        for i in range(0,(image.shape[0]-image.shape[0]%2),2):
            for j in range(0,(image.shape[1]-image.shape[1]%2),2):
                new_image[i/2,j/2] = filtered_image[i,j]
        return new_image

    def down(self, level): #upsample the image
        axis_x=self.pyramid[level].shape[0] * 2
        axis_y=self.pyramid[level].shape[1] * 2
        if len(self.img.shape) == 3 :
            new_image = np.empty((axis_x,axis_y, self.img.shape[2]))
            for depth in range(self.img.shape[2]):
                new_image[:, :, depth] = self.up_sample(self.pyramid[level][:, :, depth],(axis_x,axis_y))
            return new_image
        else:
            new_image = self.up_sample(self.pyramid[level],(axis_x,axis_y))
            return new_image
        
              
    def up(self, level): #downsample the image
        axis_x=self.pyramid[level].shape[0]//2
        axis_y=self.pyramid[level].shape[1]//2
        if len(self.img.shape) == 3 :
            new_image = np.empty((axis_x,axis_y, self.img.shape[2]))
            for depth in range(self.img.shape[2]):
                new_image[:, :, depth] = self.down_sample(self.pyramid[level][:, :, depth],(axis_x,axis_y))
            return new_image
        else:
            new_image = self.down_sample(self.pyramid[level],(axis_x,axis_y))
            return new_image
            
    def build(self):
        self.pyramid = [self.img]
        for i in range(self.levels - 1):
            self.pyramid.append(self.up(i))
    
    def get(self, level):
        if(level >=0 and level < self.levels):
            return self.pyramid[level]
        exit('Parameter error: invalid gaussian pyramid level')
        
    def show(self, name = 'gauss_pyramid'):
        for i in range(self.levels):
            cv2.imwrite(name + str(i) + '.jpg', self.get(i))
    
def load_kernels():
    kernels = []
    kernels.append(convolution.gaussian_kernel(0.3))

    return kernels

if __name__ == "__main__":
    image_names = ['input/p1-1-0.jpg', 'input/p1-1-1.jpg', 'input/p1-1-2.png', 'input/p1-1-3.png']
    name_it = 0
    kernels = load_kernels()
    pyramid_levels = 7
    for image_name in image_names:
        image = cv2.imread(image_name, 1)
        for kernel in kernels:
            pyramid = gaussian_pyramid(image, levels = pyramid_levels, kernel = kernel)
            pyramid.build()
            #plot downsample results
            print('downsample results')
            for i in range(pyramid_levels):
                cv2.imwrite('output/p1-2-2-' + str(name_it) + '.jpg', pyramid.get(i))
                print(image_name, 'output/p1-2-2-' + str(name_it) + '.jpg')
                name_it += 1

            #plot upsample results
            print('upsample results')
            for i in range(pyramid_levels):
                cv2.imwrite('output/p1-2-2-' + str(name_it) + '.jpg', pyramid.down(i))
                print(image_name, 'output/p1-2-2-' + str(name_it) + '.jpg')
                name_it += 1
