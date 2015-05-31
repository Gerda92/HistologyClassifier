#include "helpers.h"
#include "normalization.h"

clock_t then = clock();
clock_t now;

string time() {
	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	then = now;
	now = clock();
	time (&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer,80,"%d-%m-%Y %I:%M:%S",timeinfo);
	string str(buffer);
	return str;
}



int main( int argc, char** argv ) {
	
	cout<<time()<<endl;
	cout<<"Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;

	//string part = "30 Gy 2 Wo Le 2 65_part1.png";
	
	int ratios[] = {100};
	//bulkCreateRandomizedSubsets("two thirds", vector<int>(ratios, end(ratios)));

	// Get features of whole image
	int slice = 1; int p = 1;
	string newpath = get_suffix(slice, p);
	//get_ROI_features(slice, p, Rect(), "all" + newpath, "all");
	// with 
	//get_ROI_features(slice, "-SyntheticGlobal", p, "all synth" + newpath, Rect(), "all");
	//generate_random_subset("all" + newpath, "rnd all" + newpath, Rect(), 30);
	//generate_random_subset("all synth" + newpath, "rnd all synth" + newpath, Rect(), 30);
	get_ROI_features(slice, "-Norm", 1, "all norm" + newpath, Rect(), "all");
	generate_random_subset("all norm" + newpath, "rnd all norm" + newpath, Rect(), 30);
	/*
	// Color deconvolution
	//Mat img = imread("E:/DataMLMI/Slice" + to_string(slice) + "/" + part);
	//Mat img = imread("C:/Users/Gerda/Documents/MATLAB/image.png");
	Mat img = imread("C:/Users/Gerda/Desktop/image.png");

	//color_deconv(img(Rect(0, 0, 300, 300)));
	vector<Mat> deconv = color_deconv(img);
	vector<double> means, stds;
	clusters(deconv[0], means, stds);
	// END color deconvolution
	*/
	/*
	// Take features from one full slice
	int p = 1;
	string newpath = " slice " + to_string(slice) + " part " + to_string(p) + "/";
	//get_ROI_features(slice, p, Rect(), "all" + newpath, "all");
	string set = "rnd " + newpath;
	_mkdir(set.c_str());
	generate_random_subset("all" + newpath, set, Rect(), 30);

	
	int img[] = {1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1};
	
	vector<string> paths = get_paths(vector<int>(img, end(img)), "crossvalid");



	//Rect ROI = Rect(0, 0, 4300, 4236);
	// 6 images 2/3 1/3 feature extraction
	
	int p = 1;
	string newpath = " slice " + to_string(slice) + " part " + to_string(p) + "/";
	//get_ROI_features(slice, p, Rect(), "all" + newpath, "all", true, true, true, true);

	

	for (int i = 1; i < 1; i++) {
		string newpath = " slice " + to_string(slice+i) + " part " + to_string(p) + "/";
		get_ROI_features(slice+i, p, Rect(), "last third" + newpath, "last third");
		get_ROI_features(slice+i, p, Rect(), "two thirds" + newpath, "two thirds");
		
	}
	
	//ROI = Rect(0, 0, 4300, 4236);
	// extracting random pixels from 2/3
	for (int i = 0; i < 1; i++) {
		string path = "two thirds slice " + to_string(slice+i) + " part " + to_string(p) + "/";
		int ratios[] = {30};
		for (int i = 0; i < sizeof(ratios)/sizeof(*ratios); i++) {
			string set = "rnd " + path;
			_mkdir(set.c_str());
			generate_random_subset(path, set, Rect(), ratios[i]);
		}
	}
	
	// 1/3 test set features
	//ROI = Rect(4301, 0, 6660 - 4301, 4236);
	//_mkdir("set18");
	//Rect ROI = Rect(0, 0, 6660, 4236);
	//get_ROI_features(slice, part, ROI, 6, true, false, false);

	//_mkdir("rnd3");
	//generate_random_subset("set6/", "rnd3/", ROI, 3);
	//extract_patch("set6/", ROI, "test1/");
	

	*/

	cout<<time()<<endl;
	cout<<"Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;

	waitKey();

	return 0;
}