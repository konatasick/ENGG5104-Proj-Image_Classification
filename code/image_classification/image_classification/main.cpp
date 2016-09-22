#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv/cv.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "DenseFeatureExtractor.h"
#include "CodeBookGenerator.h"
#include "ImageRepresentor.h"

extern "C" {
#include <vl/generic.h>
#include <vl/fisher.h>
}
using namespace std;
using namespace cv;

int main() {
	/* Step 0: Configuration of parameters
	*/

	const int numCls = 5; // The number of classes and training/testing images per class
	const int trainImPerCls = 60;
	const int testImPerCls = 40;
	
	vector<Point2f> spmGrid; // The parameter for spatial pyramid matching
	spmGrid.push_back(Point2f(1, 1));
	spmGrid.push_back(Point2f(2, 2));
	spmGrid.push_back(Point2f(4, 4));

	const int codebookSize = 512; // The size of the codebook
	const int knn = 5; // The k-nearest neighbors (for LLC only)

	CvSVMParams params; // The parameters of SVM
    params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	//params.kernel_type = CvSVM::RBF;
    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100000, 1e-6);


	/* Step 1: Extract feature descriptors of all training samples
	*/
	cout << "Step 1: Extract feature descriptors of all training samples..." << endl;
	DenseFeatureExtractor denseExt;
	Mat samples;
	vector<Mat> trainSample;
	vector<vector<KeyPoint>> keypoints;
	Mat trainLabel;
	char clsname[5];
	char number[5];
	if (!fopen("..\\Representation\\trainRepresentation.yml", "rb")) {
		for (int cls = 0 ; cls < numCls ; ++cls) {
			for (int i = 0 ; i < trainImPerCls ; ++i) {
				cout << "SIFT descriptor: class " << cls+1 << " number " << i+1 << endl;
				sprintf(clsname, "%d", cls+1);
				sprintf(number, "%.4d", i+1);
				Mat im = imread("..\\Dataset\\Train\\"+string(clsname)+"\\image"+string(number)+".jpg", 
					CV_LOAD_IMAGE_GRAYSCALE);
				vector<KeyPoint> kps;
				Mat des = denseExt.ExtractFeature(im, kps);
				keypoints.push_back(kps);
				samples.push_back(des);
				trainSample.push_back(des);
				trainLabel.push_back(float(cls + 1));
			}
		}
		FileStorage fs("..\\Representation\\trainLabel.yml", FileStorage::WRITE);
		fs << "trainLabel" << trainLabel;
		fs.release();
	} else {
		FileStorage fs("..\\Representation\\trainLabel.yml", FileStorage::READ);
		fs["trainLabel"] >> trainLabel;
		fs.release();
	}
	cout << "done!" << endl;

	
	/* Step 2: Generate codebook
	*/
	cout << "Step 2: Generate codebook..." << endl;
	CodeBookGenerator codebookGet(codebookSize);
	Mat codebook;
	if (!fopen("..\\Codebook\\codebook.yml", "rb")) {
		codebook = codebookGet.GenerateCodebook(samples);
		FileStorage fs("..\\Codebook\\codebook.yml", FileStorage::WRITE);
		fs << "codebook" << codebook;
		fs.release();
	} else {
		FileStorage fs("..\\Codebook\\codebook.yml", FileStorage::READ);
		fs["codebook"] >> codebook;
		fs.release();
	}
	cout << "done!" << endl;

	///* Step 3.0: Test LLC
	//*/
	//cout << "Step 3.0: Test LLC" << endl;
	//ImageRepresentator imageRep(spmGrid, knn);
	//int cls = 0;
	//int i = 3 ;
	//Mat des = trainSample[cls*60+i];
	//vector<KeyPoint> kps = keypoints[cls*60+i];
	//cout << "Image representation: class " << cls+1 << " number " << i+1 << endl;
	//sprintf(clsname, "%d", cls+1);
	//sprintf(number, "%.4d", i+1);
	//Mat im = imread("..\\Dataset\\Train\\"+string(clsname)+"\\image"+string(number)+".jpg", 
	//	CV_LOAD_IMAGE_GRAYSCALE);
	//Mat imRep = imageRep.GetRepresentation(im, kps, des, codebook);
	//cout<<endl<<imRep.size()<<endl;
	//return 0;


	/* Step 3: Apply SPM and encode the feature descriptors, and then exrtact the image representation
	*/
	cout << "Step 3: Apply SPM and encode the feature descriptors, and then exrtact the image representation..." << endl;
	ImageRepresentator imageRep(spmGrid, knn);
	Mat trainRepresentation;
	if (!fopen("..\\Representation\\trainRepresentation.yml", "rb")) {
		for (int cls = 0 ; cls < numCls ; ++cls) {
			for (int i = 0 ; i < trainImPerCls ; ++i) {
				Mat des = trainSample[cls*60+i];
				vector<KeyPoint> kps = keypoints[cls*60+i];
				cout << "Image representation: class " << cls+1 << " number " << i+1 << endl;
				sprintf(clsname, "%d", cls+1);
				sprintf(number, "%.4d", i+1);
				Mat im = imread("..\\Dataset\\Train\\"+string(clsname)+"\\image"+string(number)+".jpg", 
					CV_LOAD_IMAGE_GRAYSCALE);
				Mat imRep = imageRep.GetRepresentation(im, kps, des, codebook);
				trainRepresentation.push_back(imRep);
			}
		}
		FileStorage fs("..\\Representation\\trainRepresentation.yml", FileStorage::WRITE);
		fs << "trainRepresentation" << trainRepresentation;
		fs.release();
	} else {
		FileStorage fs("..\\Representation\\trainRepresentation.yml", FileStorage::READ);
		fs["trainRepresentation"] >> trainRepresentation;
		fs.release();
	}
	cout << "done!" << endl;
	

	/* Step 4: Train linear classifier with SVM
	*/
	cout << "Step 4: Train linear classifier with SVM..." << endl;
	CvSVM classifier;
	CvParamGrid CvParamGrid_C(pow(2.0,-5), pow(2.0,15), pow(2.0,2));
	CvParamGrid CvParamGrid_gamma(pow(2.0,-15), pow(2.0,3), pow(2.0,2));
	if (!CvParamGrid_C.check() || !CvParamGrid_gamma.check())
		cout<<"The grid is NOT VALID."<<endl;
	if (!fopen("..\\Classifier\\classifier.yml", "rb")) {
		//train
		classifier.train(trainRepresentation, trainLabel, Mat(), Mat(), params);

		////autotrain
		//classifier.train_auto(trainRepresentation, trainLabel, Mat(), Mat(), params,10, CvParamGrid_C, CvParamGrid_gamma, CvSVM::get_default_grid(CvSVM::P), CvSVM::get_default_grid(CvSVM::NU), CvSVM::get_default_grid(CvSVM::COEF), CvSVM::get_default_grid(CvSVM::DEGREE), true);
		//params = classifier.get_params();
		//cout<<"gamma:"<<params.gamma<<endl;
		//cout<<"C:"<<params.C<<endl;

		classifier.save("..\\Classifier\\classifier.yml");
	} else {
		classifier.load("..\\Classifier\\classifier.yml");
	}
	cout << "done!" << endl;
	

	/* Step 5: Evaluate the classification accuracy
	*/
	cout << "Step 5: Evaluate the classification accuracy..." << endl;
	float hit = 0;
	float confusionMatrix[5][5] = {0};
	for (int cls = 0 ; cls < numCls ; ++cls) {		
		for (int i = 0 ; i < testImPerCls ; ++i) {
			cout << "Evaluate image: class " << cls+1 << " number " << i+61 << "...";
			sprintf(clsname, "%d", cls+1);
			sprintf(number, "%.4d", i+61);
			Mat im = imread("..\\Dataset\\Test\\"+string(clsname)+"\\image"+string(number)+".jpg", 
				CV_LOAD_IMAGE_GRAYSCALE);
			vector<KeyPoint> kps;
			Mat des = denseExt.ExtractFeature(im, kps);
			Mat imRep = imageRep.GetRepresentation(im, kps, des, codebook);
			int pre = classifier.predict(imRep);
			if (pre == cls+1) {
				++hit;
				cout << "hit";
			} else {
				cout << "miss";
			}
			confusionMatrix[cls][pre-1]++;
			cout << endl;
		}

	}
	cout << "Classification accuracy: " << hit/(float)(numCls*testImPerCls)<< endl;
	cout << "Confusion matrix: " <<  endl;
	for (int i = 0 ; i < numCls ; ++i) {		
		for (int j = 0 ; j < numCls ; ++j) 
			cout<<confusionMatrix[i][j]<<" ";
		cout<<endl;
	}
	
	return 0;
}
