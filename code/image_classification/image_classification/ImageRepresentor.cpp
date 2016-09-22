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

	

	Mat spmFeat;
	
	int dSize = codebook.rows;
	//////////////LLC////////////////////

	//int count = 0;
	//double weight[3] = {0.25, 0.25, 0.5};
	//
	//int nSmp = featureDescriptor.rows;
	//
 //
	//Mat idxBin(nSmp,1,CV_32F,Scalar(0));
	////Mat llc_codes = LLC_coding(codebook, featureDescriptor);
	//Mat codes = SPM_coding(codebook, featureDescriptor);

	//spmFeat = Mat(1,dSize*21,CV_32F);
	////cout<<"imrow="<<im.rows<<",imcol="<<im.cols<<endl;
	//for (int level = 0; level < spmGrid.size(); level++)
	//{
	//	int nBins = spmGrid[level].x*spmGrid[level].x;
	//	int wUnit = (int)(1/spmGrid[level].x*im.rows);
	//	int hUnit = (int)(1/spmGrid[level].y*im.cols);
	//	//cout<<"subrow="<<subrow<<",subcol="<<subcol<<endl;
	//	vector<int> idxBin(nSmp,0);
	//	Mat subFeat(nBins,dSize,CV_32F,Scalar(0));

	//	for (int i = 0; i<nSmp; i++)
	//	{
	//		int idxBin = floor(kps[i].pt.y/wUnit) + (floor(kps[i].pt.x/hUnit))*spmGrid[level].x;
	//		//cout<<level<<'\t'<<idxBin<<endl;
	//		for (int j = 0; j<dSize; j++)
	//		{
	//			float code = codes.at<float>(j,idxBin);
	//			
	//			//histogram
	//			subFeat.at<float>(idxBin,j) = subFeat.at<float>(idxBin,j)+code; 

	//			//max pooling
	// 		//	if (code > subFeat.at<float>(idxBin,j))
	//			//{
	//			//	//cout << idxBin  <<'\t'<<j<<endl;
	//			//	subFeat.at<float>(idxBin,j) = code*weight[level]; 
	//			//}
	//		}

	//	}

	//	for (int j = 0; j<nBins; j++)
	//	{
	//		subFeat.row(j).copyTo(spmFeat(Rect(count*dSize,0,dSize,1)));
	//		//cout<<subFeat.row(j)<<endl;
	//		count ++;

	//	}

	//			
	//}
	//float sum = 1e-10;

	//for (int i = 0 ; i < spmFeat.cols ; ++i) {
	//	sum += spmFeat.at<float>(0,i);	
	//}
	//for (int i = 0 ; i < spmFeat.cols ; ++i) {
	//	spmFeat.at<float>(0,i) = spmFeat.at<float>(0,i) / sum;
	//}	
 //
	//return spmFeat;


	 //End LLC

	//// SPM
	spmFeat = Mat(1,dSize*21,CV_32F);
	double weight[3] = {0.25, 0.25, 0.5};
	int count = 0;
	Mat subFeat;
	Mat zeroFeat(1,dSize,CV_32F,Scalar::all(0));

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
				subFeat.mul(weight[level]);
				subFeat.copyTo(spmFeat(Rect(count*dSize,0,dSize,1)));
				count ++;
			}
				
	}
	return spmFeat;

	
}


Mat ImageRepresentator::ComputeCode(const Mat& descriptor, const Mat& localBase) {
	//||y-Xb||^2
	//localBase = X'
	//descriptor = y'
	//code = b'
	Mat localBaseTrans = localBase.t();
	Mat code = ((localBase * localBaseTrans).inv()*localBase*(descriptor.t())).t();
	float sum = 1e-10;
	
	for (int i = 0 ; i < code.rows ; ++i) {
		sum += code.at<float>(i,0);	
	}
	for (int i = 0 ; i < code.rows ; ++i) {
		code.at<float>(i,0) = code.at<float>(i,0) / sum;
	}
	return code;
}


Mat ImageRepresentator::LLC_coding( const Mat& codebook, const Mat& featureDescriptor){
	
 
	int nbase=codebook.rows;
	int nframe=featureDescriptor.rows;
	//cout<< nframe<<','<<nbase<<endl;
 

	/////////find k nearest neighbors/////////

	cv::flann::KDTreeIndexParams indexParams(5);
	cv::flann::Index kdtree(codebook, indexParams);
	Mat indices;//(numQueries, k, CV_32S);
	Mat dists;//(numQueries, k, CV_32F);
	kdtree.knnSearch(featureDescriptor, indices, dists, knn, cv::flann::SearchParams(64));



	/////////llc approximation coding/////////
	Mat coeff(nframe, nbase,CV_32F);

	for (int i=0;i<indices.rows;i++)
	{

		Mat idx = indices.row(i);
		Mat z(knn,codebook.cols,CV_32F,Scalar(0));
		for(int col = 0 ; col < indices.cols ; col++){
			codebook.row(idx.at<int>(0,col)).copyTo(z.row(col));
			//cout<<z.row(col)<<endl;
		}

		//cout<<z<<endl;
		Mat code =  ComputeCode(z,featureDescriptor.row(i));
		//cout<<code<<endl;
		for (int j=0;j<idx.cols;j++){
			//cout<<idx.at<int>(0,j)<<endl;
			coeff.at<float>(i,idx.at<int>(0,j))=code.at<float>(0,j);
		}

	} 
 

	
	return coeff.t();
}


Mat ImageRepresentator::SPM_coding( const Mat& codebook, const Mat& featureDescriptor){
	
 
	int nbase=codebook.rows;
	int nframe=featureDescriptor.rows;
	//cout<< nframe<<','<<nbase<<endl;
 

	/////////find 1 nearest neighbors/////////
	cv::flann::KDTreeIndexParams indexParams(5);
	cv::flann::Index kdtree(codebook, indexParams);
	Mat indices;//(numQueries, k, CV_32S);
	Mat dists;//(numQueries, k, CV_32F);
	kdtree.knnSearch(featureDescriptor, indices, dists, 1, cv::flann::SearchParams(64));


	/////////spm coding/////////
	Mat coeff(nframe, nbase,CV_32F);

	for (int i=0;i<nframe;i++)
	{

		int idx = indices.at<int>(i,0);
 
		coeff.at<float>(i,idx)=1;
		 

	} 
 

	
	return coeff.t();
}