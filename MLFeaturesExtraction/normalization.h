#ifndef NORMALIZATION_H
#define NORMALIZATION_H

#include "helpers.h"
#include <Eigen/Dense>

using Eigen::Matrix3d;
using Eigen::Vector3d;

Matrix3d deconv_matrix() {
	/*
	Q << 0.65, 0.704, 0.285,
		 0.072, 0.99, 0.105,
		 0.6218, 0, 0.7831;
	*/
	Vector3d He, Eo, Bg;
	/*
	He << 0.65, 0.704, 0.285;
	Eo << 0.072, 0.99, 0.105;
	Bg << 0.6218, 0.0, 0.7831;

	He << 0.550, 0.758, 0.351;
	Eo << 0.398, 0.634, 0.600;
	Bg << 0.754, 0.077, 0.652;
	*/

	He << 0.65, 0.71, 0.26;
	Eo << 0.09, 0.95, 0.28;
	Bg << 0.63, 0.0, 0.77;

	Matrix3d Q;
	Q << He/He.norm(),
		Eo/Eo.norm(),
		Bg/Bg.norm();
	return Q;
}

Vector3d pixel_deconv(int b, int g, int r) {
	Vector3d I;
	I << -log((r+1)/255.0), -log((g+1)/255.0), -log((b+1)/255.0);
	//cout<<"I "<<I<<endl<<endl;
	//I.unaryExpr([] (double &c) {c = log10(c);});
	//cout<<"I' "<<I<<endl<<endl;

	Matrix3d Q = deconv_matrix();

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (Q(i, j) == 0) Q(i, j) = 0.001;
		}
	}

	//cout<<"Q "<<Q<<endl<<endl;
	Vector3d A = Q.inverse()*I;
	//cout<<"Q-1 "<<K.inverse()<<endl<<endl;
	//cout<<A<<endl<<endl;
	for(int i = 0; i < 3; i++) {
		if (A(i) < 0) {
			A(i) = 0;
			//cout<<"I "<<I<<endl<<endl; cout<<A<<endl<<endl;
		}
		A(i) = A(i) > 1 ? 255 : A(i)*255;
		//A(i) = exp(- (A(i) - 255) * log(255.0) / 255.0);
		//A(i) = exp(- (A(i) - 255) * log(255.0) / 255.0);
	}
	//cout<<"Col: "<<A<<endl<<endl;
	return A;
}

Vector3d pixel_conv(double he, double eo, double bg) {

	Matrix3d Q = deconv_matrix();

	Vector3d A;
	A << he, eo, bg;
	Vector3d I = -Q*A/255;
	for (int i = 0; i < 3; i++) {
		I(i) = exp(I(i))*255;
	}
	return I;
}

vector<Mat> color_deconv(Mat image) {
	vector<Mat> bgr;
	split(image, bgr);
	vector<Mat> deconv(3);
	vector<vector<Mat>> deconv_col(3, vector<Mat>(3));
	for (int i = 0; i < 3; i++) {
		deconv[i] = Mat(image.size(), CV_8UC1);
		for (int j = 0; j < 3; j++)
			deconv_col[i][j] = Mat(image.size(), CV_8UC1);
	}

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			Vector3d A = pixel_deconv((int)bgr[0].at<uchar>(i, j), (int)bgr[1].at<uchar>(i, j), (int)bgr[2].at<uchar>(i, j));
			
			//cout<<A<<endl<<endl;
			for (int c = 0; c < 3; c++) {
				deconv[c].at<uchar>(i, j) = A(c);
				/*
				for (int s = 0; s < 3; s++) {
					Vector3d X; X << 0, 0, 0;
					X(s) = A(s);
					Vector3d I = pixel_conv(X(0), X(1), X(2));
					deconv_col[s][0].at<uchar>(i, j) = I(2);
					deconv_col[s][1].at<uchar>(i, j) = I(1);
					deconv_col[s][2].at<uchar>(i, j) = I(0);
				}
				*/
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
	for (int c = 0; c < 3; c++) {
		imwrite("deconv1"+to_string(c)+".png", deconv[c]);
		Mat orig;
		merge(deconv_col[c], orig);
		imwrite("deconv_col"+to_string(c)+".png", orig);
	}
	//imwrite("orig_decon.png", orig);
	return deconv;
}

void clusters(Mat image, vector<double> &means, vector<double> &stds) {
	Mat points;
	image.reshape(1, image.rows*image.cols).convertTo(points, CV_32FC1);
	Mat labels(1, image.rows*image.cols, CV_8UC1);
	kmeans(points, 2, labels, TermCriteria( TermCriteria::EPS+TermCriteria::MAX_ITER, 100, 1.0),
		3, KMEANS_RANDOM_CENTERS, means);
	imwrite("kmeans.png", labels.reshape(1, image.rows)*255);
	stds = vector<double>(2, 0);
	//cout<<labels<<endl;
	int nones = countNonZero(labels);
	for(int i = 0; i < labels.rows; i++) {
		int label = labels.at<int>(i, 0);
		stds[label] += pow(points.at<float>(i, 0) - means[label], 2);
	}
	stds[0] = sqrt(stds[0]/(labels.rows - nones - 1));
	stds[1] = sqrt(stds[1]/(nones - 1));
	if (means[0] < means[1]) {
		swap(means[0], means[1]); swap(stds[0], stds[1]);
	}
	cout<<means[0]<<" "<<means[1]<<endl;
	cout<<stds[0]<<" "<<stds[1]<<endl;
}

double normalize(double a, double mu, double sd, double mur, double sdr) {
	return (a - mu)/sd*sdr+mur;
}

double gaussian(double x, double mu, double sigma) {
	return exp(-pow(x - mu, 2)/(2*sigma*sigma));
}

double interpolate(double a, double a_fg, double a_bg, vector<double> means, vector<double> stds) {
	//double a_fg = 
	return 0;
}




#endif