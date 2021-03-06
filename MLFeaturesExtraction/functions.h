#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <ctime>
#include <direct.h>
#include <numeric>
#include <random>

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

vector<int> feat_vect_t(Mat input);

vector<Mat> restore_feat_vect(vector<uchar> input, Rect patch);

vector<Mat> restore_feat_vect_t(vector<int> input, int nf, Rect patch);

vector<Mat> get_ROI_features(int slice, string part, Rect ROI, int set,
							 bool write_thresh = true, bool write_blur = true, bool write_orig = true);

vector<Mat> get_ROI_features(int slice, string part, Rect ROI, string path,
							 bool write_thresh = true, bool write_blur = true, bool write_orig = true);

vector<Mat> get_ROI_features(int slice, int part, Rect ROI, string path,
							 bool write_thresh = true, bool write_blur = true, bool write_orig = true);

vector<Mat> load_ROI_features(string path);

vector<int> concat_sets(vector<string> sets);

Mat load_ROI_classes(string path);

vector<int> concat_labels(vector<string> sets);

vector<vector<Mat>> restore_patches(vector<int> input, int nf, vector<Rect> patches);

vector<int> generate_random_pixels(Mat mask, float n2p);

Mat get_pixels(Mat image, vector<int> pixels);

void generate_random_subset(string path, string newpath, Rect patch, float n2p);

bool copy_file(string SRC, string DEST);

void extract_patch(string path, Rect ROI, string newpath);

vector<vector<string>> read_image_names(string path = "../ReadImageNames/example.txt");

#endif