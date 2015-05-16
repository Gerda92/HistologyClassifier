#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <ctime>
#include <direct.h>

using namespace std;
using namespace cv;

string type2str(int type);

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

std::vector<std::string> split(const std::string &s, char delim);

void calcHists(vector<Mat> channels, vector<vector<Mat>> hist, int index, Mat mask = Mat());

void plot(string window_name, vector<Mat> hist, int n, Scalar * color, int h = 250,
		  int w = 512, Point start = Point(20, 20));

void plot(string window_name, Mat image, vector<Mat> hist, int n, Scalar * color, int h = 250,
		  int w = 512, Point start = Point(20, 20), bool bgr = true, string label = "");

void plot6(string window_name, vector<vector<Mat>> hist, int n, int x, int y, Scalar * color,
	int h, int w, bool * scale, string * labels);

float * mat2arr (Mat m);

void nothing();

vector<int> feat_vect(vector<Mat> input);

vector<int> feat_vect_t(vector<Mat> input);

vector<Mat> restore_feat_vect(vector<uchar> input, Rect patch);

vector<Mat> restore_feat_vect_t(vector<int> input, int nf, Rect patch);

vector<Mat> get_ROI_features(int slice, string part, Rect ROI, int set);

vector<int> load_ROI_features(string path);

#endif