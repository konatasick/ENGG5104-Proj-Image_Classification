#include "DenseFeatureExtractor.h"


DenseFeatureExtractor::DenseFeatureExtractor(void) {
}


DenseFeatureExtractor::~DenseFeatureExtractor(void) {
}

/* ExtractFeature is for extracting feature descriptor of an image
   Input:
           im: A grayscale image in height x width.
		   keypoints: The vector for storing the key points.
   Output:
           siftMat: The SIFT descriptors of the input image.
*/
Mat DenseFeatureExtractor::ExtractFeature(const Mat& im, vector<KeyPoint>& keypoints) {
	// Initialize the feature detector and the SIFT extractor. 
	// You may choose to combine other feature descriptor.
	DenseFeatureDetector detector = DenseFeatureDetector(1.0, 1.0, 0.1, 6, 8, true, false);
	SiftDescriptorExtractor extractor;
	Mat siftMat;

	/* TODO 1: 
       Detect the feature descriptors using "detector". Place the key points in 
	   the vector "keypoints". Then use the input image to compute the descriptor
	   of each key points.
	*/

	detector.detect(im,keypoints,Mat());
	extractor.compute(im, keypoints,siftMat);

	// End TODO 1

	return siftMat;
}