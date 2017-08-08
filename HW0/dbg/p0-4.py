import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img_original = cv2.imread('../output/p0-2-b-0.jpg', False)
    img_normalized = cv2.imread('../output/p0-4-b-0.jpg', False)

    plt.figure()
    plt.hist(img_original.ravel(),256,[0,256])
    plt.show()
    #plt.savefig('original_histogram.jpg')

    plt.figure()
    plt.hist(img_normalized.ravel(),256,[0,256])
    plt.show()
    #plt.savefig('normalized_histogram.jpg')