# Video Stabilization

Video Stabilization based on keypoints matching, RANSAC and affine/projective transformation. Some details:

* fast.py: Implements FAST corner detectors.
* sift.py: Implements SIFT, note that detection module is not working appropriately while descriptor module is.
* ORB.py: Implement ORB detector, it uses fast.py as initial step and assign direction to each corner.
* matching.py: Matches two sets of keypoints based on sift descriptors and cosine distance.
* transformation.py: Applies RANSAC to find a model fitting and transforms the spaces using affine/projective transformations.
* video-stabilizaton.py: Stabilize a video using fast/ORB to detect points, SIFT/cosine distance for keypoints matching and RANSAC affine/projective transformations.
