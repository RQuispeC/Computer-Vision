import cv2
import numpy as np
import laplacian_pyramid as lp
import blending

def fromSpaceToFrequency(img):
    dft = cv2.dft(np.array(img).astype(np.float),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)    
    return np.array(cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1]))

def change_dim(polar_img):
    ans = np.empty((polar_img.shape[1], polar_img.shape[2], 2))
    for i in range(polar_img.shape[1]):
      for j in range(polar_img.shape[2]):
        ans[i,j, :] = polar_img[:, i, j]
    return ans

def change_dim_inv(polar_img):
    ans = np.empty((2, polar_img.shape[0], polar_img.shape[1]))
    for i in range(polar_img.shape[1]):
      for j in range(polar_img.shape[2]):
        ans[:, i, j] = polar_img[i, j, :]
    return ans

def fromFrequencyToSpace(polar_img):
    dft = np.array(cv2.polarToCart(polar_img[0, :, :], polar_img[1, :, :]))
    dft = change_dim(dft)
    dft_ishift = np.fft.ifftshift(dft)
    recovered = cv2.idft(dft_ishift)
    recovered = cv2.magnitude(recovered[:,:,0], recovered[:,:,1])
    return (recovered*255.0)/recovered.max() #normalized

def zeroid(img_or, compl):
    for i in range(img_or.shape[0]):
      for j in range(img_or.shape[1]):
        if img_or[i, j] == 0.0:
          compl[i, j] = 0.0;
    return compl

def lower_values(img, selection, isPercentage = True):
    sorted_img = np.sort(np.unique(img))
    #print('shape', sorted_img.shape,sorted_img.min(), sorted_img.max())
    #print ((int)(len(sorted_img)*selection))
    if not isPercentage:
        lim = sorted_img[selection]
    else:
        lim = sorted_img[min((int)(len(sorted_img)*selection), len(sorted_img) -1)]
    #print('lim', lim)    
    img[img > lim] = 0
    return img
 
def upper_values(img, selection, isPercentage = True):
    sorted_img = np.sort(np.unique(img))[::-1]
    #print('shape', sorted_img.shape,sorted_img.min(), sorted_img.max())
    #print ((int)(len(sorted_img)*selection))
    if not isPercentage:
        lim = sorted_img[selection]
    else:
        lim = sorted_img[min((int)(len(sorted_img)*selection), len(sorted_img) -1)]
    #print('lim', lim)    
    img[img < lim] = 0
    return img
    
def frec_blending(fimg_first, fimg_second, strategy = 'left_right', width = 80, slide = 20):
    rows, cols = fimg_first.shape[1:]
    crow,ccol = rows//2 , cols//2
    if strategy == 'left_right':
        fimg_first[:, :, :ccol] = fimg_second[:, :, :ccol]
    elif strategy == 'bottom_up':
        fimg_first[:, :crow, :] = fimg_second[:, :crow, :]
    elif strategy == 'centered':
        fimg_first[:, crow-width:crow+width, ccol-width:ccol+width] = fimg_second[:, crow-width:crow+width, ccol-width:ccol+width]
    elif strategy == 'chessboard':    
        for i in range(rows):
            flag = (i % 2 == 0)
            for j in range(cols):
                if flag:
                    fimg_first[:, i, j] = fimg_second[:, i, j]
                flag = not flag
    elif strategy == 'sliding_columns':
        flag = True
        for j in range(0, cols, slide):
            if flag:
                fimg_first[:, :, j:min(cols, j+slide)] = fimg_second[:, :, j:min(cols, j+slide)]
            flag = not flag
    elif strategy == 'sliding_rows':
        flag = True
        for i in range(0, rows, slide):
            if flag:
                fimg_first[:, i:min(rows, i+slide), :] = fimg_second[:, i:min(rows, i+slide), :]
            flag = not flag
    else:
        exit('Error: not valid frequency blending strategy')
    return fimg_first

if __name__ == "__main__":
    image_names = ['input/p1-1-2.png', 'input/p1-1-3.png', 'input/p1-1-0.jpg', 'input/p1-1-1.jpg']
    name_it = 0

    
    flag_values = [False, True, True, True, True]
    selection_values = [1, 0.25, 0.5, 0.75, 1.0]
    print('Exploring Fourier Space')
    for image_name in image_names:
        image = cv2.imread(image_name, 0)
        frec_image = fromSpaceToFrequency(image)
        #low values in phase
        print('low values in magnitud')
        for flag, selection in zip(flag_values, selection_values):
            frec_tmp = frec_image.copy()
            frec_tmp[0, :, :] = lower_values(frec_tmp[0, :, :], selection, flag)
            cv2.imwrite('output/p1-3-1-' + str(name_it) + '.jpg', fromFrequencyToSpace(frec_tmp))
            print(image_name, 'output/p1-3-1-' + str(name_it) + '.jpg')
            name_it += 1

        #high values in phase
        print('high values in magnitud')
        for flag, selection in zip(flag_values, selection_values):
            frec_tmp = frec_image.copy()
            frec_tmp[0, :, :] = upper_values(frec_tmp[0, :, :], selection, flag)
            cv2.imwrite('output/p1-3-1-' + str(name_it) + '.jpg', fromFrequencyToSpace(frec_tmp))
            print(image_name, 'output/p1-3-1-' + str(name_it) + '.jpg')
            name_it += 1
        #low values in magnitude
        print('low values in phase')
        for flag, selection in zip(flag_values, selection_values):
            frec_tmp = frec_image.copy()
            frec_tmp[1, :, :] = lower_values(frec_tmp[1, :, :], selection, flag)
            cv2.imwrite('output/p1-3-1-' + str(name_it) + '.jpg', fromFrequencyToSpace(frec_tmp))
            print(image_name, 'output/p1-3-1-' + str(name_it) + '.jpg')
            name_it += 1
        #high values in magnitude
        print('high values in phase')
        for flag, selection in zip(flag_values, selection_values):
            frec_tmp = frec_image.copy()
            frec_tmp[1, :, :] = upper_values(frec_tmp[1, :, :], selection, flag)
            cv2.imwrite('output/p1-3-1-' + str(name_it) + '.jpg', fromFrequencyToSpace(frec_tmp))
            print(image_name, 'output/p1-3-1-' + str(name_it) + '.jpg')
            name_it += 1
        

    print('Frequency Blending')
    blend_strategies = ['left_right', 'bottom_up', 'centered', 'chessboard', 'sliding_rows', 'sliding_columns']
    for i in range(3):
        image2 = cv2.imread(image_names[i], 0)
        image1 = cv2.imread(image_names[i+1], 0)
        image2 = cv2.resize(image2, (image1.shape[1],image1.shape[0]))
        mask = blending.set_mask(mask_type = 'left_right', img_shape = image1.shape, value = 1)
        image1 = image1 * mask
        image2 = image2 * blending.inverse(mask, value = 1)
        image1[image1 == 0] = 128
        image2[image2 == 0] = 128
        frec_image1 = fromSpaceToFrequency(image1)
        frec_image2 = fromSpaceToFrequency(image2)
        for blend_strategy in blend_strategies:
            frec_blend_img = frec_blending(frec_image1.copy(), frec_image2.copy(), strategy = blend_strategy)
            cv2.imwrite('output/p1-3-2-' + str(name_it) + '.jpg', fromFrequencyToSpace(frec_blend_img))
            print(image_names[i], image_names[i+1], blend_strategy, 'output/p1-3-2-' + str(name_it) + '.jpg')
            name_it += 1
