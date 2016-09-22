#include "CodeBookGenerator.h"


CodeBookGenerator::CodeBookGenerator(const int codebookSize) {
	this->codebookSize = codebookSize;
}


CodeBookGenerator::~CodeBookGenerator(void) {
}


/* GenerateCodebook is for generating the codebook with the feature descriptors
   Input:
           samples: The feature descriptors.
   Output:
           codebook: The codebook generated. It is used for encoding feature 
					 descriptors.
*/
Mat CodeBookGenerator::GenerateCodebook(const Mat& descriptors) {
	int sampleNum = descriptors.rows;
	int dims = descriptors.cols;
	Mat labels(sampleNum, 1, CV_32FC1);
	Mat codebook(this->codebookSize, dims, CV_32FC1);

	/* TODO 2: 
       Use kmeans or fisher vector encoding to generate the codebook
	*/


	//TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	BOWKMeansTrainer bowTrainer(codebookSize);//,tc);//,1,KMEANS_PP_CENTERS);
	codebook=bowTrainer.cluster(descriptors);   


	// End TODO 2

	return codebook;
}