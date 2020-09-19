#include "libs.h"
#include "cuda\gpu_debayer.h"

using namespace cv;

int main()
{
    auto path = "F:\\Workdatas\\debayer\\raw.bmp";


    auto raw = imread(path,0);
    int rows = raw.rows;
    int cols = raw.cols;

    Mat imcv(rows, cols, CV_8UC3);
   
    cvtColor(raw, imcv, COLOR_BayerBG2BGR);

    // 
    unsigned char *buf_raw_host, *buf_raw_dev;
    unsigned char *buf_rgb_host, *buf_rgb_host2, *buf_rgb_dev;

    auto curtn = cudaMalloc(&buf_raw_dev, rows * cols);
    assert(curtn == cudaSuccess);

    curtn = cudaMalloc(&buf_rgb_dev, rows * cols * 3);
    assert(curtn == cudaSuccess);

    curtn = cudaMallocHost(&buf_rgb_host, rows * cols * 3);
    assert(curtn == cudaSuccess);

    curtn = cudaMallocHost(&buf_rgb_host2, rows * cols * 3);
    assert(curtn == cudaSuccess);

    curtn = cudaMemcpy(buf_raw_dev, raw.data, rows * cols, cudaMemcpyHostToDevice);
    assert(curtn == cudaSuccess);

    gpu_bayer_to_rgb_n3(buf_raw_dev, cols, rows, buf_rgb_dev, nullptr);
    auto cuerr = cudaGetLastError();

    curtn = cudaMemcpy(buf_rgb_host, buf_rgb_dev, rows * cols * 3, cudaMemcpyDeviceToHost);
    assert(curtn == cudaSuccess);

    Mat rgb(rows, cols, CV_8UC3, buf_rgb_host);
    cvtColor(rgb, rgb, COLOR_RGB2BGR);


    // old 
    gpu_bayer_to_rgb(buf_raw_dev, cols, rows, buf_rgb_dev);
    cuerr = cudaGetLastError();

    curtn = cudaMemcpy(buf_rgb_host, buf_rgb_dev, rows * cols * 3, cudaMemcpyDeviceToHost);
    assert(curtn == cudaSuccess);

    Mat rgb2(rows, cols, CV_8UC3, buf_rgb_host);
    cvtColor(rgb2, rgb2, COLOR_RGB2BGR);
    // end old

    imshow("opencv", imcv);
    imshow("new", rgb);
    imshow("old", rgb2);

    waitKey();

    destroyAllWindows();





    return 0;
}