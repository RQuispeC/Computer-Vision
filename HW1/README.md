# Image Blending

Blending is divided in spatial and frequency domain, some details:

* gaussian\_pyramid.py: Implements a class for gaussian pyramid (really!), it works great the only detail to consider is that the number of levels must make sense with the size of input image.
* laplacian\_pyramid.py: Uses gaussian\_pyramid.py to implements a class for laplacian pyramid (yes, really!), also consider the number of levels.
* convolution.py: Some basic convolution operation.
* blending.py: Implements blending in spatial domain, the results are better with a greater number of pyramids levels.
* fourier.py: Implements som expermients for blending in frequency domain, note that this is impossible.
