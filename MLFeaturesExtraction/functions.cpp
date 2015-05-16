#include "functions.h"

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

vector<int> load_ROI_features(string path) {
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
		features.push_back(img);
	}
	ffile.close();
	return feat_vect(features);
}

vector<Mat> get_ROI_features(int slice, string part, Rect ROI, int set) {

	Mat src, hsv, mask;
	
	src = imread("E:/DataMLMI/Slice" + to_string(slice) + "/" + part + ".png", 1);
	mask = imread("E:/DataMLMI/Slice" + to_string(slice) + "/Labels_" + part + ".png", 0);
	src = src(ROI);

	vector<Mat> orig_features;
	split(src, orig_features);

	cvtColor(src, hsv, COLOR_BGR2HSV);
	vector<Mat> hsv_features;
	split(hsv, hsv_features);

	orig_features.insert(orig_features.end(), hsv_features.begin(), hsv_features.end());

	ofstream ffile;

	ffile.open("set" + to_string(set) + "/features.txt");
	ffile << slice << endl << part << endl;
	ffile << ROI.x << ' ' << ROI.y << ' ' << ROI.width << ' ' << ROI.height <<endl;

	vector<Mat> features(orig_features.begin(), orig_features.end());
	const string of[] = {"b", "g", "r", "h", "s", "v"};
	for (int i = 0; i < features.size(); i++) {
		imwrite("set" + to_string(set) + "/" + of[i] + ".png", features[i]);
		ffile << of[i] << endl;
	}

	int kernels[] = {11};
	int nkernels = sizeof(kernels)/sizeof(int);

	//features.reserve(nkernels*orig_features.size());

	for (int f = 0; f < orig_features.size(); f++) {
		for (int i = 0; i < nkernels; i++) {
			Mat filtered(src.size(), CV_8UC1);
			int index = f*nkernels + i;
			medianBlur(orig_features[f], filtered, kernels[i]);
			features.push_back(filtered);
			//imshow("W", filtered);
			string fID = of[f] + ",median," + to_string(kernels[i]);
			imwrite("set" + to_string(set) + "/" + fID + ".png", filtered);
			ffile << fID <<endl;
		}
	}

	// detecting red channel high
	Mat thrsh;
	inRange(orig_features[3], 130, 142, thrsh);
	string fID = of[3] + ",hist,gauss,";
	GaussianBlur(thrsh, thrsh, Size(21, 21), 20);
	//features.push_back(thrsh);
	imwrite("set" + to_string(set) + "/" + fID + ".png", thrsh);
	ffile << fID <<endl;

	ffile.close();

	return features;
}