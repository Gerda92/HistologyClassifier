#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "helpers.h"
#include <Eigen/Dense>

using Eigen::Matrix3d;
using Eigen::Vector3d;

Vector3d pixel_deconv(int b, int g, int r) {
	Vector3d I;
	I << -log10(r/256.0), -log10(g/256.0), -log10(b/256.0);
	//cout<<"I "<<I<<endl<<endl;
	//I.unaryExpr([] (double &c) {c = log10(c);});
	//cout<<"I' "<<I<<endl<<endl;
	/*
	Q << 0.65, 0.704, 0.285,
		 0.072, 0.99, 0.105,
		 0.6218, 0, 0.7831;
	*/
	Vector3d He, Eo, Bg;
	He << 0.65, 0.704, 0.285;
	Eo << 0.072, 0.99, 0.105;
	Bg << 0.6218, 0.0, 0.7831;

	/*
	He << 0.550, 0.758, 0.351;
	Eo << 0.398, 0.634, 0.600;
	Bg << 0.754, 0.077, 0.652;
	*/
	Matrix3d Q;
	Q << He/He.norm(),
		Eo/Eo.norm(),
		Bg/Bg.norm();
	Matrix3d K;
	K << He,
		Eo,
		Bg;
	//cout<<"K "<<K.transpose()<<endl<<endl;
	Vector3d A = Q.inverse()*I;
	//cout<<"Q-1 "<<K.inverse()<<endl<<endl;
	//cout<<A<<endl<<endl;
	for(int i = 0; i < 2; i++) if (A(i) < 0) {
		A(i) = 0;
		//cout<<"I "<<I<<endl<<endl; cout<<A<<endl<<endl;
	}
	return A*255/A.sum();
}

vector<Mat> color_deconv(Mat image) {
	vector<Mat> bgr;
	split(image, bgr);
	vector<Mat> deconv(3);
	for (int i = 0; i < 3; i++) deconv[i] = Mat(image.size(), CV_8UC1);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			Vector3d A = pixel_deconv((int)bgr[0].at<uchar>(i, j), (int)bgr[1].at<uchar>(i, j), (int)bgr[2].at<uchar>(i, j));
			//cout<<A<<endl<<endl;
			for (int c = 0; c < 3; c++) {
				deconv[c].at<uchar>(i, j) = A(c);
				//cout<<A(c)<<" ";
				//cout<<(int)deconv[c].at<uchar>(i, j)<<" ";
			}
			//cout<<endl;
		}
		//cout<<"Row "<<i<<endl;
	}
	//cout<<deconv[0]<<endl;
	//cout<<deconv[1]<<endl;
	//cout<<deconv[2]<<endl;
	for (int c = 0; c < 3; c++) imwrite("deconv1"+to_string(c)+".png", deconv[c]);
	return deconv;
}

#endif