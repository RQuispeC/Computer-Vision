import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import pylab

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

def fromFrequencyToSpace(polar_img):
    dft = np.array(cv2.polarToCart(polar_img[0, :, :], polar_img[1, :, :]))
    dft = change_dim(dft)
    dft_shift = np.fft.ifftshift(dft)
    recovered = cv2.idft(dft_shift)
    recovered = cv2.magnitude(recovered[:,:,0], recovered[:,:,1])
    return (recovered*255.0)/recovered.max()

def zeroid(img_or, compl):
    for i in range(img_or.shape[0]):
      for j in range(img_or.shape[1]):
        if img_or[i, j] == 0.0:
          compl[i, j] = 0.0;
    return compl

def lower(img, selection, isPercentage = True):
    sorted_img = np.sort(np.unique(img))
    print('shape', sorted_img.shape,sorted_img.min(), sorted_img.max())
    print ((int)(len(sorted_img)*selection))
    if not isPercentage:
        lim = sorted_img[selection]
    else:
        lim = sorted_img[min((int)(len(sorted_img)*selection), len(sorted_img) -1)]
    print('lim', lim)    
    img[img > lim] = 0
    return img
def comple(mask):
    print(np.ones(mask.shape) - mask,mask,mask.shape)
    return np.ones(mask.shape) - mask
    
def create_mask(image):
    return np.hstack((np.zeros((image.shape[0], image.shape[1]//2)), np.ones((image.shape[0], image.shape[1] - image.shape[1]//2))))
    
def upper(img, selection, isPercentage = True):
    sorted_img = np.sort(np.unique(img))
    print('shape', sorted_img.shape,sorted_img.min(), sorted_img.max())
    print ((int)(len(sorted_img)*selection))
    if not isPercentage:
        lim = sorted_img[max(len(sorted_img) - selection, 0)]
    else:
        lim = sorted_img[max(len(sorted_img) - min((int)(len(sorted_img)*selection), len(sorted_img) -1), 0)]
    print('lim', lim)    
    img[img < lim] = 0
    return img
    
def combine(fimg_first, fimg_second):
    print(fimg_first.shape)
    rows, cols = fimg_first.shape[1:]
    crow,ccol = rows//2 , cols//2
    width = 80
    #fimg_first[:, crow-width:crow+width, ccol-width:ccol+width] = fimg_second[:, crow-width:crow+width, ccol-width:ccol+width]
    fimg_first[:, :crow, :cols] = fimg_second[:, :crow, :cols]
    #fimg_first[:, :rows, :ccol] = fimg_second[:, :rows, :ccol]
    '''
    for i in range(rows):
      flag = (i % 2 == 0)
      for j in range(cols):
        if flag:
          fimg_first[:, i, j] = fimg_second[:, i, j]
        flag = not flag
    '''
    slide = 50
    for j in range(cols, slide):
      fimg_first[:, :, j:min(cols, j+slide)] = fimg_second[:, :, j:min(cols, j+slide)]
    return fimg_first
    

if __name__ == "__main__":
    file_name_input2 = 'adin.jpg'
    file_name_input1 = 'flaquita.jpg'
    image1 = cv2.imread(file_name_input1,0)
    image2 = cv2.imread(file_name_input2,0)
    mask = create_mask(image1)
    image1 = image1 * mask
    image2 = image2 * comple(mask)
    image1[image1 == 0] = 128
    image2[image2 == 0] = 128
    frec_image1 = fromSpaceToFrequency(image1)
    frec_image2 = fromSpaceToFrequency(image2)
    cv2.imwrite('frec_blending5.jpg', fromFrequencyToSpace(combine(frec_image1, frec_image2)))
    #frec_image1[0, :, :] = lower(frec_image1[0, :, :], 1  )
    #frec_image1[1, :, :] = zeroid(frec_image1[0, :, :], frec_image1[1, :, :])
    #frec_image1[1, :, :] = lower(frec_image1[1, :, :], 0.3)
    #ans = fromFrequencyToSpace(frec_image1)
    #cv2.imwrite('lower_2x.jpg', ans)

 
    
    
