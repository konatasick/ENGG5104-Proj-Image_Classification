#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace std;
using namespace cv;

#pragma once
class ImageRepresentator
{
public:
	ImageRepresentator(const vector<Point2f>& spmGrid, const int knn);
	~ImageRepresentator(void);
	Mat GetRepresentation(const Mat& im, const vector<KeyPoint>& kps, 
		const Mat& featureDescriptor, const Mat& codebook);
	
private:
	Mat ComputeCode(const Mat& descriptor, const Mat& localBase);
	Mat ImageRepresentator::LLC_coding(const Mat& feaSet, const Mat& codebook);
	Mat ImageRepresentator::SPM_coding(const Mat& feaSet, const Mat& codebook);
	vector<Point2f> spmGrid;
	int knn;
};

