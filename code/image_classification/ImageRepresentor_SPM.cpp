#include "ImageRepresentor.h"


ImageRepresentator::ImageRepresentator(const vector<Point2f>& spmGrid, const int knn) {
	this->spmGrid = spmGrid;
	this->knn = knn;
}


ImageRepresentator::~ImageRepresentator(void) {
}


/* ImageRepresentator is for extracting image representation of an image
   Input:
           im: A grayscale image in height x width.
		   kps: The vector which has stored the key points.
		   featureDescriptor: The feature descriptors of the input image.
		   codebook: The codebook for encoding the feature descriptors.
   Output:
           spmFeat: The final representation of the input image.
*/
Mat ImageRepresentator::GetRepresentation(const Mat& im, const vector<KeyPoint>& kps,
	const Mat& featureDescriptor, const Mat& codebook) {


	Mat spmFeat(1,21504,CV_32F);


	/* TODO 3: 
       Apply spatial pyramid matching to group the descriptors according to their 
	   coordinates. The configuration of the local grids is stored in the member
	   variable "spmGrid" of this class.
	*/
	
	double weight[3] = {0.25, 0.25, 0.5};
	int count = 0;
	Mat subFeat;
	Mat zeroFeat(1,1024,CV_32F,Scalar::all(0));


	// End TODO 3


	/* TODO 4: 
       You may have three choices. Apply (1) descriptor quantization; (2) approximated 
	   LLC or (3) Fisher Vector encoding.
	*/


    
    //create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    //create Sift descriptor extractor
    Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);    
    //create Sift feature point extracter
    Ptr<FeatureDetector> detector(new SiftFeatureDetector());

    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(codebook);     
	//cout<<"imrow="<<im.rows<<",imcol="<<im.cols<<endl;
	for (int level = 0; level < spmGrid.size(); level++)
	{
		int subrow = (int)1/spmGrid[level].x*im.rows;
		int subcol = (int)1/spmGrid[level].y*im.cols;
		//cout<<"subrow="<<subrow<<",subcol="<<subcol<<endl;

		for (int i = 0; i < spmGrid[level].x; i++)
			for (int j = 0; j < spmGrid[level].y; j++)
			{
				Mat subim = im(Rect(i*subcol,j*subrow,subcol,subrow));
				vector<KeyPoint> keypoints;
				detector->detect(subim,keypoints);
				//extract BoW (or BoF) descriptor from given image
				
				bowDE.compute(subim,keypoints,subFeat);	
				if (subFeat.rows == 0)
					subFeat = zeroFeat;
				subFeat.copyTo(spmFeat(Rect(count*1024,0,1024,1)));
				count ++;

	// End TODO 4


	/* TODO 5: 
       You should implement the pooling method to aggregate the encoded descriptor within 
	   each local grid set by SPM. You should concatenate the aggregated vectors of all 
	   local grids as the final image representation. Please store the representation in
	   "spmFeat".
	*/



			}
				
	}


	// End TODO 5

	

	return spmFeat;
}


Mat ImageRepresentator::ComputeCode(const Mat& descriptor, const Mat& localBase) {
	Mat localBaseTrans = localBase.t();
	Mat code = ((localBase * localBaseTrans).inv()*localBase*(descriptor.t())).t();
	float sum = 1e-10;
	for (int i = 0 ; i < code.cols ; ++i) {
		sum += code.at<float>(0, i);	
	}
	for (int i = 0 ; i < code.cols ; ++i) {
		code.at<float>(0, i) = code.at<float>(0, i) / sum;
	}
	return code;
}