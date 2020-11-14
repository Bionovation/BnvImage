#include "libs.h"
#include <iostream>
#include <vector>

using namespace cv;

void usm(Mat in, Mat &out)
{
	cv::GaussianBlur(in, out, cv::Size(0, 0), 3);
	cv::addWeighted(in, 1.5, out, -0.5, 0, out);
}

int main() {
	String path = R"(E:\Workdatas\enhance\src\2020-05-26-09-19-00-1_652210_9494_72_67.jpg)";
	auto img = imread(path, 1);
	imshow("src", img);

	Mat hls;
	cvtColor(img, hls, CV_BGR2HLS);
	//imshow("hls", hls);

	std::vector<Mat> hlsarr;
	split(hls, hlsarr);
	auto L = hlsarr[1];

	// usm hls
	Mat usmL;
	usm(L, usmL);

	hlsarr[1] = usmL;

	merge(hlsarr, hls);

	Mat rgb;
	cvtColor(hls, rgb, CV_HLS2BGR);
	imshow("usm-hls", rgb);

	// usm-rgb
	std::vector<Mat> rgbarr;
	split(img, rgbarr);

	Mat usmR, usmG, usmB;
	usm(rgbarr[0], usmB);
	usm(rgbarr[1], usmG);
	usm(rgbarr[2], usmR);

	rgbarr[0] = usmB;
	rgbarr[1] = usmG;
	rgbarr[2] = usmR;

	Mat usmrgb;
	merge(rgbarr, usmrgb);
	imshow("usm-rgb", usmrgb);

	auto err = usmrgb - rgb;
	double maxvalue = 0;
	cv::minMaxIdx(err, 0, &maxvalue);
	

	waitKey();
	destroyAllWindows();
	return 0;
}