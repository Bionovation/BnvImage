#include "libs.h"

using namespace cv;

int main()
{
    auto path = "F:\\Workdatas\\debayer\\raw.bmp";
    auto im = imread(path,0);

    Mat rgb(im.rows, im.cols, CV_8UC3);
    cvtColor(im, rgb, COLOR_BayerBG2BGR);

    imshow("d", rgb);
    waitKey();

    destroyAllWindows();





    return 0;
}