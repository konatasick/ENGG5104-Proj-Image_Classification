#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

#pragma once
class DenseFeatureExtractor
{
public:
	DenseFeatureExtractor(void);
	~DenseFeatureExtractor(void);
	Mat ExtractFeature(const Mat& im, vector<KeyPoint>& keypoints);
};

