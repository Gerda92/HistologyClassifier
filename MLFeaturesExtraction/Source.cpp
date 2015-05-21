#include "functions.h"

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

	int slice = 1;
	string part = "30 Gy 2 Wo Le 2 65_part1";

	//Rect ROI = Rect(400, 900, 200, 250);
	//Rect ROI = Rect(0, 0, 6660, 4236);
	Rect rects[] = {Rect(420, 970, 180, 150), Rect(1380, 1470, 200, 190), Rect(1950, 1870, 100, 100)};
	/*
	int offset = 3;
	for (int i = 0; i < sizeof(rects)/sizeof(*rects); i++) {
		string set = "set" + to_string(i + offset) + "/";
		mkdir(set.c_str());
		get_ROI_features(slice, part, rects[i], i+offset);
	}
	*/
	const char * sets[] = {"set3/", "set4/", "set5/"};
	vector<string> vsets(sets, end(sets));
	vector<int> features = concat_sets(vsets);

	

	//vector<Mat> features = load_ROI_features(set);

	/*
	Mat tf = features[0].reshape(1, features[0].rows*features[0].cols);
	for(int i = 1; i < features.size(); i++) {
		Mat f = features[i].reshape(1, features[0].rows*features[0].cols);
		hconcat(tf, f, tf);
	}

	tf.convertTo(tf, CV_32FC1);

	Mat classes = load_ROI_classes(set).reshape(1,50000);
	
	classes.convertTo(classes, CV_32SC1);
    RTrees::Params  params( 4, // max_depth,
                        500, // min_sample_count,
                        0, // regression_accuracy,
                        false, // use_surrogates,
                        2, // max_categories,
                        Mat(), // priors,
                        false, // calc_var_importance,
                        3, // nactive_vars,
                        TermCriteria(TermCriteria::MAX_ITER, 5, 0) // max_num_of_trees_in_the_forest,
                       );

    Ptr<RTrees> rtrees = StatModel::train<RTrees>(tf, ROW_SAMPLE, classes, params);
	*/
//	int i;
//	cin>>i;

	/*
	vector<int> v = load_ROI_features("set1/");

	vector<int> v2 = feat_vect_t(features);
	restore_feat_vect_t(v2, features.size(), ROI);
	
	int example = 10;
	for(int i = 0; i < features.size(); i++) {
		cout<<v[ROI.width*ROI.height*i + example]<<endl;
	}

	for(int i = 0; i < features.size(); i++) {
		cout<<v2[features.size()*example + i]<<endl;
	}
	
	
	int i;
	cin >> i;
	*/
	/*
	Mat src, hsv, mask;
	
	src = imread("E:/DataMLMI/Slice" + to_string(slice) + "/" + part + ".png", 1);
	mask = imread("E:/DataMLMI/Slice" + to_string(slice) + "/Labels_" + part + ".png", 0);
	//src = src(Rect(300, 400, 1000, 1000));

	cout<<time()<<endl;
	cout<<"Src and mask read. Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;

	vector<Mat> orig_features;
	split(src, orig_features);

	cvtColor(src, hsv, COLOR_BGR2HSV);
	vector<Mat> hsv_features;
	split(src, hsv_features);

	cout<<time()<<endl;
	cout<<"BGR-HSV cvt; splits made. Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;

	orig_features.insert(orig_features.end(), hsv_features.begin(), hsv_features.end());

	cout<<time()<<endl;
	cout<<"Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;

	ofstream ffile;

	ffile.open("fearures.txt", ios_base::app);

	vector<Mat> features(orig_features.begin(), orig_features.end());
	const string of[] = {"b", "g", "r", "h", "s", "v"};
	for (int i = 0; i < features.size(); i++) {
		imwrite(part + "," + of[i] + ".png", features[i]);
	}

	int kernels[] = {11};
	int nkernels = sizeof(kernels)/sizeof(int);

	features.reserve(nkernels*orig_features.size());

	for (int f = 0; f < orig_features.size(); f++) {
		for (int i = 0; i < nkernels; i++) {
			Mat filtered(src.size(), CV_8UC1);
			int index = f*nkernels + i;
			medianBlur(orig_features[f], filtered, kernels[i]);
			features[index] = filtered;
			//imshow("W", filtered);
			imwrite(part + "," + of[i] + ".png", features[i]);
		}
	}

	ffile.close()

	cout<<time()<<endl;
	cout<<"Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;
	*/

	/*
	
	Mat pmask = mask(patch);
	imwrite("set1/fl.png", pmask);
		ofstream myfile;
	myfile.open ("set1/fd.txt");
	myfile << "Per. of positive: " << countNonZero(pmask)*1.0/pmask.rows/pmask.cols<<endl;
	myfile.close();
	for (int i = 0; i < features.size(); i++) {
		features[i] = imread("Window " + to_string(i) + ".png");
		Mat crop = features[i](patch);
		imwrite("set1/f" + to_string(i) + ".png", crop);
	}
	cout<<"Done."<<endl;
	*/

	/*
	for (int i = 0; i < features.size(); i++) {
		features[i] = imread("set1/f" + to_string(i) + ".png", 0);
	}
	vector<uchar> v = feat_vect(features);
	cout<<time()<<endl;
	cout<<"Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;

	vector<Mat> nf(12, Mat(patch.size(), CV_8UC1));
	nf = restore_feat_vect(v, patch);

	for (int i = 0; i < nf.size(); i++) {
		imshow("W " + to_string(i), nf[i]);
	}
	
	cout<<time()<<endl;
	cout<<"Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;
	waitKey();
	*/
	/*
	
	cout<<time()<<endl;
	cout<<"Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;

	vector <float> out = feat_vect(features);

	cout<<time()<<endl;
	cout<<"Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;

	//serialize(out);
	/*
	cout<<time()<<endl;
	cout<<"Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;

	vector<float> v = load_serialized();

	vector<Mat> features = restore_feat_vect(v);

	for (int i = 0; i < features.size(); i++) {
		imwrite("Restored Window " + to_string(i) + ".png", features[i]);
	}
	*/
//	waitKey();

	return 0;
}