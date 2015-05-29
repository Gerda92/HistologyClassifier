#ifndef HELPERS_H
#define HELPERS_H

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

string timet() {
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	time (&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer,80,"%d-%m-%Y %I:%M:%S",timeinfo);
	string str(buffer);
	return str;
}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}


string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

vector<int> feat_vect_t(vector<Mat> input) {
	vector<int> output(input[0].rows*input[0].cols*input.size());
	for (int f = 0; f < input.size(); f++) {
		for (int i = 0; i < input[f].rows; i++) {
			for (int j = 0; j < input[f].cols; j++) {
				int example = i*input[f].cols+j;
				int index = example*input.size()+f;
				output[index] = (int)input[f].at<uchar>(i,j);
			}
		}
	}
	return output;
}

vector<int> feat_vect_t(Mat input) {
	vector<Mat> v(1, input);
	return feat_vect_t(v);
}

vector<Mat> restore_feat_vect_t(vector<int> input, int nf, Rect patch) {
	vector<Mat> features(nf, Mat(Size(patch.height, patch.width), CV_8UC1));
	for (int f = 0; f < nf; f++) {
		vector<int> v(patch.height*patch.width);
		for (int i = 0; i < patch.height*patch.width; i++) {
			v[i] = input[nf*i+f];
		}
		Mat m(v);
		m = m.reshape(1, patch.height);
		imwrite("s" + to_string(f)+".png", m);
		features[f] = m.clone();
	}
	return features;
}
vector<vector<Mat>> restore_patches(vector<int> input, int nf, vector<Rect> patches) {
	vector<vector<Mat>> v;
	_mkdir("Test restore/");
	int offset = 0;
	for (int i = 0; i < patches.size(); i++) {
		vector<int> set(input.begin()+offset, input.end());
		v.push_back(restore_feat_vect_t(set, nf, patches[i]));
		for(int j = 0; j < v[i].size(); j++) {
			imwrite("Test restore/" + to_string(i) + to_string(j) + ".png", v[i][j]);
		}
		offset += patches[i].width*patches[i].height*nf;
	}
	return v;
}

vector<Mat> restore_feat_vect(vector<int> input, Rect patch) {
	vector<Mat> features(12, Mat(Size(patch.height, patch.width), CV_8UC1));
	for (int f = 0; f < 12; f++) {
		vector<uchar>vec(input.begin() + f*patch.height*patch.width, input.begin() + (f+1)*patch.height*patch.width);
		Mat m(vec);
		m = m.reshape(1, patch.height);
		//imshow("s" + to_string(f), m);
		features[f] = m.clone();
	}
	return features;
}

vector<string> read_feature_file(string path) {
	ifstream ffile;
	ffile.open(path + "features.txt");
	int slice; string part; int x, y, w, h; string f;
	ffile>>slice; getline(ffile, f); getline(ffile, part); ffile>>x>>y>>w>>h; getline(ffile, f);
	//cout<<slice<<endl<<part<<endl;
	vector<string> files;
	while(getline(ffile,f)) {
		files.push_back(f);
	}
	ffile.close();
	return files;
}

vector<Mat> load_ROI_features(string path) {
	vector<string> images = read_feature_file(path);
	vector<Mat> features(images.size());
	for(int i = 0; i < images.size(); i++) {
		cout<<images[i]<<endl;
		Mat img = imread(path+images[i]+".png", 0);
		//imshow(f, img);
		cout<<img.rows<<" "<<img.cols<<endl;
		features[i] = img;
	}
	return features;
}

Mat load_ROI_classes(string path) {
	return imread(path+"fl.png", 0);
}

bool copy_file(string SRC, string DEST)
{
    std::ifstream src(SRC, std::ios::binary);
    std::ofstream dest(DEST, std::ios::binary);
    dest << src.rdbuf();
    return src && dest;
}
vector<int> concat_sets(vector<string> sets) {
	vector<int> f;
	for (int i = 0; i < sets.size(); i++) {
		vector<int> fi = feat_vect_t(load_ROI_features(sets[i]));
		//restore_feat_vect_t(fi,19,);
		cout<<fi.size()<<endl;
		f.insert(f.end(), fi.begin(), fi.end());
	}
	return f;
}

vector<int> concat_labels(vector<string> sets) {
	vector<int> f;
	for (int i = 0; i < sets.size(); i++) {
		vector<int> fi = feat_vect_t(load_ROI_classes(sets[i]));
		f.insert(f.end(), fi.begin(), fi.end());
	}
	return f;
}

vector<int> generate_random_pixels(Mat mask, float n2p) {
	Mat labels = mask.reshape(1, 1);
	int positives = countNonZero(labels);
	int n = labels.cols;
	int pos_to_sample = positives/200*200;
	int neg_to_sample = pos_to_sample*n2p/200*200;
	vector<int> perm(n - positives);
	iota(perm.begin(), perm.end(), 0);
	unsigned seed = time(0);
	shuffle(perm.begin(), perm.end(), std::default_random_engine(seed));

	vector<int> positions(n - positives);
	int current = 0;
	vector<int> pixels(neg_to_sample + pos_to_sample);
	for (int i = 0; i < n; i++) {
		if ((int)labels.at<uchar>(0, i) > 0) {
			if (i - current < pos_to_sample)
				pixels[i - current] = i;
			continue;
		}
		positions[current] = i; current++;
	}
	for (int i = 0; i < neg_to_sample; i++) {
		pixels[pos_to_sample + i] = positions[perm[i]];
	}
	/*
	Mat random(image.size(), CV_8UC1);
	for (int i = 0; i < neg_to_sample; i++) {
		int index = positions[perm[i]];
		int row = index/image.cols; int col = index % image.cols;
		circle(random, Point(col, row), 0.5, Scalar(0, 0, 0), -1);
	}
	imwrite("rand.png", random);
	*/
	return pixels;
}

Mat get_pixels(Mat image, vector<int> pixels) {
	Mat samples = image.reshape(1, 1);
	Mat sampled(1, pixels.size(), CV_8UC1, Scalar(0, 0, 0));
	for (int i = 0; i < pixels.size(); i++) {
		uchar a = samples.at<uchar>(0, pixels[i]);
		sampled.at<uchar>(0, i) = a;
	}
	sampled = sampled.reshape(1, 200);
	imwrite("sampled.png", sampled);
	return sampled;
}

void generate_random_subset(string path, string newpath, Rect patch, float n2p) {
	_mkdir(newpath.c_str());
	vector<Mat> features = load_ROI_features(path);
	Mat classes = load_ROI_classes(path);
	if (patch == Rect()) patch = Rect(0, 0, classes.cols, classes.rows); 
	vector<int> rand = generate_random_pixels(classes(patch).clone(), n2p);
	vector<string> files = read_feature_file(path);
	copy_file(path+"features.txt", newpath+"features.txt");
	for (int i = 0; i < features.size(); i++) {
		Mat fi = get_pixels(features[i](patch).clone(), rand);
		imwrite(newpath+files[i]+".png", fi);
	}
	//Mat white(100, rand.size()/200, CV_8UC1, Scalar(255, 255, 255));
	//Mat black(100, rand.size()/200, CV_8UC1, Scalar(0, 0, 0));
	Mat label = get_pixels(classes(patch).clone(), rand);
	//vconcat(white, black, label);
	imwrite(newpath+"fl.png", label);
}

void extract_patch(string path, Rect ROI, string newpath) {
	vector<Mat> f = load_ROI_features(path);
	vector<string> files = read_feature_file(path);
	copy_file(path+"features.txt", newpath+"features.txt");
	for (int i = 0; i < files.size(); i++) {
		imwrite(newpath + files[i] + ".png", f[i](ROI));
	}
	Mat c = load_ROI_classes(path);
	imwrite(newpath + "fl.png", c(ROI));
}

vector<vector<string>> read_image_names(string path = "../ReadImageNames/example.txt") {
	vector<vector<string>> images(16);
	ifstream input(path);
	for (int s = 0; s <16; s++) {
		string line;
		getline(input, line);
		vector<string> files = split(line, '\'');
		images[s] = vector<string>(files.begin()+2, files.end());
	}
	return images;
}

vector<vector<int>> bulkLoadVectorize(vector<string> paths, string fov) {
	vector<vector<int>> v(paths.size());
	for (int i = 0; i < paths.size(); i++) {
		cout<<"Loading "<<fov<<" "<<paths[i]<<endl;
		if (fov == "features")
			v[i] = loadFeaturesFast(paths[i]);
		else
			v[i] = feat_vect_t(load_ROI_classes(paths[i]));
		cout<<"Size: "<<v[i].size()<<endl;
	}
	return v;
}


vector<string> get_paths(vector<int> images, string base_path) {
	vector<string> paths(images.size()/2);
	for (int i = 0; i < images.size(); i+=2) {
		paths[i/2] = base_path + " slice " + to_string(images[i]) + " part " + to_string(images[i+1]) + "/";
	}
	return paths;
}

vector<int> exclude_set(vector<vector<int>> totrain, int n) {
	int size = 0;
	for(vector<int> v : totrain) size += v.size();
	cout<<"Total size: "<<size<<" . Excluding set of size "<<totrain[n].size()<<".\n";
	vector<int> out (size - totrain[n].size());
	int index = 0;
	for(int i = 0; i < totrain.size(); i++)
		if (i != n)
			for (int j = 0; j < totrain[i].size(); i++) {
				out[index] = totrain[i][j];
				index++;
			}
	cout<<size - totrain[n].size()<<" = "<<out.size()<<endl;
	return out;
}

void precomputeROI(string path) {
	vector<Mat> m = load_ROI_features(path);
	Mat result(m[0].rows, m[0].cols*m.size(), CV_8UC1);
	for (int i = 0; i < m.size(); i++) {
		for (int c = 0; c < m[0].cols; c++) {
			result.col(c*m.size()+i) = (m[i].col(c) + 0);
		}
	}
	imwrite(path + "precomp.png", result);
}

vector<int> loadFeaturesFast(string path) {
	cout<<timet()<<endl;
	Mat f = imread(path + "precomp.png", 0);
	cout<<timet()<<endl;
	vector<int> v;
	f.reshape(1, 1).row(0).copyTo(v);
	cout<<timet()<<endl;
	return v;
}

#endif