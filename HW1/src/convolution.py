import numpy as np
import cv2

def gaussian_kernel(gauss_kernel_par):
    vect = np.array([1/4.0 - gauss_kernel_par/2.0, 1/4.0, gauss_kernel_par, 1/4.0, 1/4.0 - gauss_kernel_par/2.0])
    return np.outer(vect, np.transpose(vect))

def padding(img, padding_width, padding_type = 'mirror', padding_color = 0):
    if padding_type == 'zero' or padding_type == 'constant':
        img = np.pad(img, padding_width, mode = 'constant', constant_values = padding_color)
    elif padding_type == 'mirror':
        img = np.pad(img, padding_width, mode = 'reflect')
    else:
        exit('Invalid argument: padding type')
    return img

def flip(kernel):
    return np.flip(np.flip(kernel, 0), 1)

def validate(img_shape, kernel_shape):
    if img_shape[0] < kernel_shape[0] or img_shape[1] < kernel_shape[1]:
        exit('Error: kernel must be smaller')
    if kernel_shape[0] % 2 == 0 or kernel_shape[1] % 2 == 0 or kernel_shape[0] != kernel_shape[1]:
        exit('Error: kernel dimensions must be odd and equal')
    return True

def norm(img):
    img /= img.max()
    img = (img * 255).astype("uint8")
    return img

def convolve(img_input, kernel, padding_type = 'zero', padding_color = 0, normalize = False):
    convolved_img = np.empty(img_input.shape)
    if len(img_input.shape) == 3:
        img_input[:, :, 0] = convolve(img_input[:, :, 0], kernel, padding_type, padding_color, normalize)
        img_input[:, :, 1] = convolve(img_input[:, :, 1], kernel, padding_type, padding_color, normalize)
        img_input[:, :, 2] = convolve(img_input[:, :, 2], kernel, padding_type, padding_color, normalize)
        return img_input
    
    #    validate data
    validate(img_input.shape, kernel.shape)
    #   flit kernel
    kernel = flip(kernel)
    #    padding
    img = padding(img_input, kernel.shape[0]//2, padding_type, padding_color=padding_color)
    #    colvolve
    space = kernel.shape[0]//2
    for row in range(space,convolved_img.shape[0]+space):
        for col in range(space, convolved_img.shape[1]+space): 
            aux_matrix = img[row-space:row+space + 1,col-space:col+space + 1]
            new_value = (int)((aux_matrix*kernel).sum())
            convolved_img[row-space,col-space]=new_value

    if normalize == True:
        convolved_img = norm(convolved_img)
    return convolved_img

def load_kernels():
    kernels = []
    kernels.append(convolution.gaussian_kernel(0.3))
    return kernels

if __name__ == "__main__":
    image_names = ['input/p1-1-0.jpg', 'input/p1-1-1.jpg', 'input/p1-1-2.png', 'input/p1-1-3.png']
    kernels = load_kernels()
    name_it = 0
    for image_name in image_names:
        image = cv2.imread(image_name, 0)
        for kernel in kernels:
            ans=convolve(image, kernel, padding_type = 'zero')
            cv2.imwrite('output/p1-2-1-' + str(name_it) + '.jpg', ans)
            print(image_name, 'output/p1-2-1-' + str(name_it) + '.jpg')
            ans=convolve(image, kernel, padding_type = 'constant', padding_color = 128)
            cv2.imwrite('output/p1-2-1-' + str(name_it+1) + '.jpg', ans)
            print(image_name, 'output/p1-2-1-' + str(name_it+1) + '.jpg')
            ans=convolve(image, kernel, padding_type = 'constant', padding_color = 255)
            cv2.imwrite('output/p1-2-1-' + str(name_it+2) + '.jpg', ans)
            print(image_name, 'output/p1-2-1-' + str(name_it+2) + '.jpg')
            ans=convolve(image, kernel, padding_type = 'mirror')
            cv2.imwrite('output/p1-2-1-' + str(name_it+3) + '.jpg', ans)
            print(image_name, 'output/p1-2-1-' + str(name_it+3) + '.jpg')
            name_it += 4




