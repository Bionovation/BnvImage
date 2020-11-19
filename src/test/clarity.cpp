#include "libs.h"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;


float clarity(Mat im)
{
    resize(im, im, Size(im.cols / 2, im.rows / 2));
    Sobel(im, im, CV_8U, 1, 0);
    return (float)mean(im)[0];
}

int main()
{
    auto folder = R"(F:\Workdatas\FocusAlg\pathology-sy\5\)";

    for (int i = 0; i <= 200; i += 2)
    {
        auto fn = folder + to_string(i) + ".jpg";
        auto im = imread(fn, 0);
        if (im.empty()) {
            continue;
        }
        auto v = clarity(im);
        cout << fn<< "  " << v << endl;
    }

    cin.ignore();
    return 0;
}