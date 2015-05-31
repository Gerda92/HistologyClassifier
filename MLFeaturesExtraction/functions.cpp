#include "functions.h"

void calcHists(vector<Mat> channels, vector<vector<Mat>> hist, int index, Mat mask) {
	int histSize = 256;
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	for (int i = 0; i < channels.size(); i++) {
		calcHist( &channels[i], 1, 0, Mat(), hist[index][i], 1, &histSize, &histRange);
	}
} 

void plot(string window_name, vector<Mat> hist, int n, Scalar * color, int h, int w, Point start) {
	Mat image(h, w, CV_8UC3, Scalar( 0,0,0) );
	
	plot(window_name, image, hist, n, color, h, w, start);

	namedWindow(window_name, WINDOW_NORMAL);
	imshow(window_name, image);
}

void plot(string window_name, Mat image, vector<Mat> hist, int n, Scalar * color, int h, int w, Point start, bool bgr, string label) {
	double hn = 0; double minVal; double maxVal; 
	for (int s = 0; s < hist.size(); s++) {
		minMaxLoc(hist[s], &minVal, &maxVal);
		if (maxVal > hn) hn = maxVal;
	}
	int interval = cvRound( (double) w/n);

	int offset = 70; int lline = 5;

	putText(image, label, start + Point(0,20),
		FONT_HERSHEY_COMPLEX, 1, Scalar(250,250,250), 1);

	for (int s = 0; s < hist.size(); s++) {
		for (int i = 1; i < n; i++) {
			line(image,
				start + Point(interval*(i-1), h - offset - cvRound(hist[s].at<float>(i-1))/hn*(h - offset*1.5)),
				start + Point(interval*(i), h - offset - cvRound(hist[s].at<float>(i))/hn*(h - offset*1.5)),
							color[s], 2);
			if (i%20 == 0) {
				line(image, start + Point(interval*(i-1), h - (int)offset*0.4 - lline),
					start + Point(interval*(i-1), h - (int)offset*0.4 - 5*lline), Scalar(250,250,250));
				putText(image, to_string(i), start + Point(interval*(i-1) + lline, h - (int)offset*0.4), 
					FONT_HERSHEY_COMPLEX_SMALL, 1.2, Scalar(250,250,250), 1);
			}
		}
	}

	for (int i = 0; i < 256; i++) {
		if (!bgr) {
			Mat seg(20, interval, CV_8UC3, Scalar(i,255,255));
			cvtColor(seg, seg, COLOR_HSV2BGR);
			if (i >= 180) break;
			Rect roi(start + Point(i*interval, h), seg.size());
			seg.copyTo(image(roi));
		} else {
			Mat seg(20, interval, CV_8UC3, Scalar(0,0,i));
			Rect roi(start + Point(i*interval, h), seg.size());
			seg.copyTo(image(roi));		
		}
	}



}

void plot6(string image_name, vector<vector<Mat>> hist, int n, int x, int y, Scalar * color,
	int h, int w, bool * scale, string * labels) {
	Point global_offset = Point(20, 20);
	int offset_x = 50; int offset_y = 50;
	Mat image((h + offset_y)*y, (w + offset_x)*x, CV_8UC3, Scalar(0,0,0));
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			plot(image_name, image, hist[i*y+j], n, color,
				h, w, global_offset + Point(i*(w + offset_x), j*(h + offset_y)), scale[i*y+j], labels[i*y+j]);
		}
	}
	imwrite("C:/Users/Gerda/Desktop/Graphs/" + image_name + ".jpg", image);
	//namedWindow(window_name, WINDOW_KEEPRATIO);
	//imshow(window_name, image);
}

float * mat2arr (Mat m) {
	float* arr = new float(m.cols);
	for (int i = 0; i < m.cols; i++)
		arr[i] = m.at<float>(i);
	return arr;
}

void nothing() {

	/*
	ifstream input("../ReadImageNames/example.txt");

	for (int s = 1; s <=16; s++) {
		string line;
		getline(input, line);
		vector<string> files = split(line, '\'');
		for (int j = 2; j < files.size(); j++) {
			Mat src, hsv, mask, imask;
			src = imread("E:/DataMLMI/Slice" + to_string(s) + "/" + files[j]);
			mask = imread("E:/DataMLMI/GTSlice" + to_string(s) + "/Labels_" + files[j], 0);
			if (!src.data || !mask.data) {
				cout<<files[j]<<endl;
				waitKey(0);
			} else {
				cout << files[j] << " read." << endl;
			}

			bitwise_not (mask, imask);

			vector<Mat> bgr_planes;
			split(src, bgr_planes);

			cvtColor(src, hsv, COLOR_BGR2HSV);
			vector<Mat> hsv_planes;
			split(hsv, hsv_planes);

			vector<vector<Mat>> hist(6, vector<Mat>(bgr_planes.size()));

			Scalar cl[6] = {Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255),
				Scalar(255, 0, 255), Scalar(255, 255, 0), Scalar(0, 255, 255)};	

			int histSize = 256;
			float range[] = { 0, 256 } ;
			const float* histRange = { range };

			//calcHists(bgr_planes, hist, 0);

			for (int i = 0; i < bgr_planes.size(); i++) {
				calcHist( &bgr_planes[i], 1, 0, Mat(), hist[0][i], 1, &histSize, &histRange);
			}

			for (int i = 0; i < bgr_planes.size(); i++) {
				calcHist( &bgr_planes[i], 1, 0, imask, hist[1][i], 1, &histSize, &histRange);
			}

			for (int i = 0; i < bgr_planes.size(); i++) {
				calcHist( &bgr_planes[i], 1, 0, mask, hist[2][i], 1, &histSize, &histRange);
			}

			for (int i = 0; i < hsv_planes.size(); i++) {
				calcHist( &hsv_planes[i], 1, 0, Mat(), hist[3][i], 1, &histSize, &histRange);
			}

			for (int i = 0; i < hsv_planes.size(); i++) {
				calcHist( &hsv_planes[i], 1, 0, imask, hist[4][i], 1, &histSize, &histRange);
			}

			for (int i = 0; i < hsv_planes.size(); i++) {
				calcHist( &hsv_planes[i], 1, 0, mask, hist[5][i], 1, &histSize, &histRange);
			}

			bool scale[6] = {true, true, true, false, false, false};
			string labels[] = {"RGB - all", "RGB - negative", "RGB - positive",
				"HSV - all", "HSV - negative", "HSV - positive"};

			plot6("Slice" + to_string(s) + files[j], hist, histSize, 2, 3, cl, 500, 1024, scale, labels);
			
			cout << files[j] << " : histogram created." << endl;

		}
	}

	*/

	Mat src, hsv;

	/// Load image
	src = imread("C:/Users/Gerda/Desktop/cat.jpg", 1 );

	if( !src.data ) {

	}

	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	cvtColor(src, hsv, COLOR_BGR2HSV);
	vector<Mat> hsv_planes;
	split(hsv, hsv_planes);

	for (int i = 0; i < hsv_planes.size(); i++)
		bgr_planes.push_back(hsv_planes[i]);

	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	vector<Mat> hist(bgr_planes.size());

	/// Compute the histograms:

	int hist_w = 512; int hist_h = 500;
	int bin_w = cvRound( (double) hist_w/histSize );

	for (int i = 0; i < bgr_planes.size(); i++) {
		calcHist( &bgr_planes[i], 1, 0, Mat(), hist[i], 1, &histSize, &histRange, uniform, accumulate );
		normalize(hist[i], hist[i], 0, hist_h, NORM_MINMAX, -1, Mat() );
	}

	// Draw the histograms for B, G and R

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	Scalar cl[6] = {Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255),
		Scalar(255, 0, 255), Scalar(255, 255, 0), Scalar(0, 255, 255)};	

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ ) {
		for (int c = 0; c < bgr_planes.size(); c++)
		  line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist[c].at<float>(i-1)) ) ,
						   Point( bin_w*(i), hist_h - cvRound(hist[c].at<float>(i)) ),
						   cl[c], 2, 8, 0  );
	}

	/// Display
	namedWindow("RGB Histogram", WINDOW_NORMAL);
	imshow("RGB Histogram", histImage );

	//vector<int*> 

	for( int i = 1; i < histSize; i++ ) {
		for (int c = 0; c < bgr_planes.size(); c++)
		  line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist[c].at<float>(i-1)) ) ,
						   Point( bin_w*(i), hist_h - cvRound(hist[c].at<float>(i)) ),
						   cl[c], 2, 8, 0  );
	}	
		/*
	vector<Mat> hist(bgr_planes.size());
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	Scalar cl[6] = {Scalar(255, 0, 0), Scalar(0, 255, 0), Scalar(0, 0, 255),
		Scalar(255, 0, 255), Scalar(255, 255, 0), Scalar(0, 255, 255)};	

	Mat imask;

	bitwise_not (mask, imask);

	//imshow("some an", imask);

	for (int i = 0; i < bgr_planes.size(); i++) {
		calcHist( &bgr_planes[i], 1, 0, Mat(), hist[i], 1, &histSize, &histRange, true, false);
		//normalize(hist[i], hist[i], 0, hist_h, NORM_MINMAX, -1, Mat() );
	}

	plot("RGB", hist, histSize, cl);

	for (int i = 0; i < hsv_planes.size(); i++) {
		calcHist( &hsv_planes[i], 1, 0, Mat(), hist[i], 1, &histSize, &histRange, true, false);
		//normalize(hist[i], hist[i], 0, hist_h, NORM_MINMAX, -1, Mat() );
	}

	plot("HSV", hist, histSize, cl);

	for (int i = 0; i < bgr_planes.size(); i++) {
		calcHist( &bgr_planes[i], 1, 0, imask, hist[i], 1, &histSize, &histRange, true, false);
		//normalize(hist[i], hist[i], 0, hist_h, NORM_MINMAX, -1, Mat() );
	}

	plot("RGB - neg", hist, histSize, cl);

	for (int i = 0; i < hsv_planes.size(); i++) {
		calcHist( &hsv_planes[i], 1, 0, imask, hist[i], 1, &histSize, &histRange, true, false);
		//normalize(hist[i], hist[i], 0, hist_h, NORM_MINMAX, -1, Mat() );
	}

	plot("HSV - neg", hist, histSize, cl);

	for (int i = 0; i < bgr_planes.size(); i++) {
		calcHist( &bgr_planes[i], 1, 0, mask, hist[i], 1, &histSize, &histRange, true, false);
		//normalize(hist[i], hist[i], 0, hist_h, NORM_MINMAX, -1, Mat() );
	}

	plot("RGB - with mask", hist, histSize, cl);

	for (int i = 0; i < hsv_planes.size(); i++) {
		calcHist( &hsv_planes[i], 1, 0, mask, hist[i], 1, &histSize, &histRange, true, false);
		//normalize(hist[i], hist[i], 0, hist_h, NORM_MINMAX, -1, Mat() );
	}

	plot("HSV - with mask", hist, histSize, cl);

	for (int i = 0; i < hsv_planes.size(); i++)
		bgr_planes.push_back(hsv_planes[i]);

	*/


	/*
	int myints[] = {255, 0, 255, 0, 1, 0};
	vector<int> v (myints, myints + sizeof(myints) / sizeof(int) );
	Mat m(v);
	m.convertTo(m, CV_8UC1);
	cout<<m;
	namedWindow("f", WINDOW_NORMAL);
	imshow("f", m);
	waitKey();
	return 0;
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

	ROI = Rect(0, 0, 6660, 4236);
	_mkdir("set16");
	get_ROI_features(slice, part, ROI, 16, true, false, false);
	//Rect ROI = Rect(0, 0, 6660, 4236);
	//get_ROI_features(slice, part, ROI, 6, true, false, false);

	


	Mat mask, image;
	image = imread("E:/DataMLMI/Slice" + to_string(slice) + "/" + part + ".png", 0);
	mask = imread("E:/DataMLMI/GTSlice" + to_string(slice) + "/Labels_" + part + ".png", 0);
	vector<int> rand = generate_random_pixels(mask);
	get_pixels(image, rand);
	*/
	/*

	//vector<Mat> features = load_ROI_features(set);

	
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
	
		vector<vector<int>> features =  bulkLoadVectorize(paths, "features");
	vector<vector<int>> labels = bulkLoadVectorize(paths, "labels");

	cout<<time()<<endl;
	cout<<"Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;

	//vector<string> paths_test = get_paths(vector<int>(img, end(img)), "crossvalid");

	//vector<vector<int>> features_test = bulkLoadVectorize(paths_test, "features");
	//vector<vector<int>> labels_test = bulkLoadVectorize(paths_test, "labels");

	cout<<time()<<endl;
	cout<<"Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;
	// cross validation
	for (int i = 0; i < paths.size(); i++) {
		vector<int> feat = exclude_set(features, i);
		vector<int> lab = exclude_set(labels, i);
		// train
		//vector<int> featt = features_test[i];
		//vector<int> labt = labels_test[i];
		// test
		cout<<time()<<endl;
		cout<<"Secs elapsed: "<<double(now - then) / CLOCKS_PER_SEC<<endl;
	}
	

	// Fetures for small patch
	//Rect ROI = Rect(400, 900, 200, 250);
	
	_mkdir("set11");
	get_ROI_features(slice, part, ROI, 11);

	// Features for three small selected patches
	Rect rects[] = {Rect(420, 970, 180, 150), Rect(1380, 1470, 200, 190), Rect(1950, 1870, 100, 100)};
	vector<Rect> patches(rects, end(rects));

	int offset = 13;
	for (int i = 0; i < sizeof(rects)/sizeof(*rects); i++) {
		string set = "set" + to_string(i + offset) + "/";
		_mkdir(set.c_str());
		get_ROI_features(slice, part, rects[i], i+offset);
	}
	
	
	const char * sets[] = {"set3/", "set4/", "set5/"};
	vector<string> vsets(sets, end(sets));
	vector<int> features = concat_sets(vsets);
	vector<int> classes = concat_labels(vsets);
	restore_patches(features, 19, patches);
	*/


}

void serialize (vector<float> v) {
	ofstream myfile;
	myfile.open ("fvector.txt");
	for (int i = 0; i < v.size(); i++) {
		myfile << v[i] << ' ';
	}
	myfile.close();
}

vector<float> load_serialized () {
	ifstream myfile;
	myfile.open ("fvector.txt");
	string line;
	getline(myfile, line);
	std::stringstream ss(line);
	vector<float> elems;
    float item;
    while (ss >> item) {
        elems.push_back(item);
    }
	myfile.close();
	return elems;
}

vector<int> feat_vect(vector<Mat> input) {
	vector<int> output;
	output.reserve(input[0].rows*input[0].cols*input.size());
	for (int f = 0; f < input.size(); f++) {
		vector<uchar>vec(input[0].rows*input[0].cols);
		for (int i = 0; i < input[f].rows; i++) {
			for (int j = 0; j < input[f].cols; j++) {
				int index = input[f].cols*i +j;
				vec[index] = (int)input[f].at<uchar>(i,j);
			}
		}
		output.insert(output.end(), vec.begin(), vec.end());

	}
	return output;
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
	ifstream ffile;

	ffile.open(path + "features.txt");
	int slice; string part; int x, y, w, h; string f;
	ffile>>slice; getline(ffile, f); getline(ffile, part); ffile>>x>>y>>w>>h; getline(ffile, f);
	//cout<<slice<<endl<<part<<endl;
	vector<Mat> features;
	while(getline(ffile,f)) {
		//cout<<f<<endl;
		Mat img = imread(path+f+".png", 0);
		//imshow(f, img);
		//cout<<img.rows<<" "<<img.cols<<endl;
		features.push_back(img);
	}
	ffile.close();
	return features;
}

Mat load_ROI_classes(string path) {
	return imread(path+"fl.png", 0);
}

vector<Mat> get_ROI_features(int slice, string part, Rect ROI, int set,
							 bool write_thresh, bool write_blur, bool write_orig) {
	return get_ROI_features(slice, part, ROI, "set" + to_string(set) + "/",
		write_thresh, write_blur, write_orig);
}

vector<Mat> get_ROI_features(int slice, int part, Rect ROI, string path,
							 bool write_thresh, bool write_blur, bool write_orig) {
	vector<vector<string>> names = read_image_names();
	return get_ROI_features(slice, names[slice-1][part-1], ROI, path,
		write_thresh, write_blur, write_orig);
}

vector<Mat> get_ROI_features(int slice, string part, Rect ROI, string path,
							 bool write_thresh, bool write_blur, bool write_orig) {

	_mkdir(path.c_str());
	Mat src, hsv, mask;
	
	src = imread("E:/DataMLMI/Slice" + to_string(slice) + "/" + part, 1);
	mask = imread("E:/DataMLMI/GTSlice" + to_string(slice) + "/Labels_" + part, 0);
	if (ROI.width == 0) ROI = Rect(0, 0, src.cols, src.rows);
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

bool copy_file(string SRC, string DEST)
{
    std::ifstream src(SRC, std::ios::binary);
    std::ofstream dest(DEST, std::ios::binary);
    dest << src.rdbuf();
    return src && dest;
}