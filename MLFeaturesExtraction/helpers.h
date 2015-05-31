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
	if (neg_to_sample >= n - positives) {
		vector<int> pixels(n/200*200);
		iota(pixels.begin(), pixels.end(), 0);
		return pixels;
	}
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

vector<int> loadFeaturesFast(string path) {
	cout<<timet()<<endl;
	Mat f = imread(path + "precomp.png", 0);
	cout<<timet()<<endl;
	vector<int> v;
	f.reshape(1, 1).row(0).copyTo(v);
	cout<<timet()<<endl;
	return v;
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

string get_suffix(int slice, int part) {
	return " slice " + to_string(slice) + " part " + to_string(part) + "/";
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


vector<Mat> get_ROI_features(int slice, string suffix, string part,  string path, Rect ROI = Rect(), string twothirds = "ROI",
							 bool write_thresh = true, bool write_blur = true, bool write_orig = true) {

	_mkdir(path.c_str());
	Mat src, hsv, mask;
	
	src = imread("E:/DataMLMI/Slice" + to_string(slice) + suffix + "/" + part, 1);
	cout<<"E:/DataMLMI/Slice" + to_string(slice) + suffix + "/" + part<<endl;
	mask = imread("E:/DataMLMI/GTSlice" + to_string(slice) + "/Labels_" + part, 0);
	if (twothirds == "two thirds") ROI = Rect(0, 0, src.cols*2/3, src.rows);
	else if (twothirds == "last third") ROI = Rect(src.cols*2/3+1, 0, src.cols - src.cols*2/3 - 1, src.rows);
	else if (ROI.width == 0) ROI = Rect(0, 0, src.cols, src.rows);
	imwrite(path + "fl.png", mask(ROI));
	src = src(ROI);

	vector<Mat> orig_features;
	split(src, orig_features);

	cvtColor(src, hsv, COLOR_BGR2HSV);
	vector<Mat> hsv_features;
	split(hsv, hsv_features);

	orig_features.insert(orig_features.end(), hsv_features.begin(), hsv_features.end());

	ofstream ffile;

	ffile.open(path + "features.txt");
	ffile << slice << endl << part << endl;
	ffile << ROI.x << ' ' << ROI.y << ' ' << ROI.width << ' ' << ROI.height <<endl;

	vector<Mat> features(orig_features.begin(), orig_features.end());
	const string of[] = {"b", "g", "r", "h", "s", "v"};
	for (int i = 0; i < features.size(); i++) {
		if (write_orig)
			imwrite(path + of[i] + ".png", features[i]);
		ffile << of[i] << endl;
	}

	int kernels[] = {11, 17};
	int nkernels = sizeof(kernels)/sizeof(int);

	//features.reserve(nkernels*orig_features.size());
	string blurs[] = {"median"};
	int nblurs = sizeof(blurs)/sizeof(*blurs);

	for (int b = 0; b < nblurs; b++)
		for (int f = 0; f < orig_features.size(); f++) {
			for (int i = 0; i < nkernels; i++) {
				Mat filtered(src.size(), CV_8UC1);
				int index = f*nkernels + i;
				string fID = of[f] + "," + blurs[b] + "," + to_string(kernels[i]);
				if (write_blur) {
					if (blurs[b] == "median")
						medianBlur(orig_features[f], filtered, kernels[i]);
					if (blurs[b] == "gaussian")
						GaussianBlur(orig_features[f], filtered, Size(31, 31), kernels[i]);
					//imshow("W", filtered);
					imwrite(path + fID + ".png", filtered);
				}
				features.push_back(filtered);
				ffile << fID <<endl;
			}
		}

	// detecting hue
	int channel = 3; // Hue
	Mat thrsh;
	int ranges[] = {130, 140, 135, 140, 135, 142, 130, 139};
	int nranges = sizeof(ranges)/sizeof(int);
	int size = 31;
	int sigmas[] = {10};
	int nsigmas = sizeof(sigmas)/sizeof(int);
	for (int s = 0; s < nsigmas; s++) {
		for (int i = 0; i < nranges; i+=2) {
			inRange(orig_features[channel], ranges[i], ranges[i+1], thrsh);
			string fID = of[channel] + ",hist," + to_string(ranges[i]) + ","
				+ to_string(ranges[i+1]) + ",gauss," + to_string(sigmas[s]);
			GaussianBlur(thrsh, thrsh, Size(size, size), sigmas[s]);
			imwrite(path + fID + ".png", thrsh);
			features.push_back(thrsh);
			ffile << fID <<endl;
		}
	}
	ffile.close();

	return features;
}

vector<Mat> get_ROI_features(int slice, string part, string path, Rect ROI = Rect(), string twothirds = "ROI",
							 bool write_thresh = true, bool write_blur = true, bool write_orig = true) {
	return get_ROI_features(slice, "", part, path, ROI, twothirds, write_thresh, write_blur, write_orig);
}

vector<Mat> get_ROI_features(int slice, string part, Rect ROI, int set, string twothirds = "ROI",
							 bool write_thresh = true, bool write_blur = true, bool write_orig = true) {
	return get_ROI_features(slice, part, "set" + to_string(set) + "/", ROI, twothirds,
		write_thresh, write_blur, write_orig);
}

vector<Mat> get_ROI_features(int slice, string suffix, int part, string path, Rect ROI = Rect(), string twothirds = "ROI",
							 bool write_thresh = true, bool write_blur = true, bool write_orig = true) {
	vector<vector<string>> names = read_image_names();
	return get_ROI_features(slice, suffix, names[slice-1][part-1], path, ROI, twothirds,
		write_thresh, write_blur, write_orig);
}

vector<Mat> get_ROI_features(int slice, int part, string path, Rect ROI = Rect(), string twothirds = "ROI",
							 bool write_thresh = true, bool write_blur = true, bool write_orig = true) {
	return get_ROI_features(slice, "", part, path, ROI, twothirds, write_thresh, write_blur, write_orig);
}

// Creating randomized subsets of slices
void bulkCreateRandomizedSubsets(string base, vector<int>ratios, int nslices = 6) {
	int p = 1; // part
	for (int i = 1; i <= nslices; i++) {
		string path = base + " slice " + to_string(i) + " part " + to_string(p) + "/";
		for (int j = 0; j < ratios.size(); j++) {
			string set = "rnd " + to_string(ratios[j]) + " " + path;
			//_mkdir(set.c_str());
			generate_random_subset(path, set, Rect(), ratios[j]);
			precomputeROI(set);
		}
	}
}


#endif