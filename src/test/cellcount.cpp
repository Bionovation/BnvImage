#include "libs.h"
#include "../cuda/gpu_debayer.h"

#include <iostream>
#include <string>

#include "utils/FindFiles.h"

using namespace std;
using namespace cv;



#pragma region 小徐的计算红细胞数量算法

struct coord_t {
    int x;
    int y;
    int width;
    int height;
};


const float PI = 3.1415926;
const float r = 15;



void findContours2( Mat& src, vector<vector<Point>>& contours, vector<Vec4i>& hierarchy,
    int retr = RETR_LIST, int method = CHAIN_APPROX_SIMPLE, Point offset = Point(0, 0))
{
    CvMat c_image = CvMat(src);
    MemStorage storage(cvCreateMemStorage());
    CvSeq* _ccontours = 0;
    cvFindContours(&c_image, storage, &_ccontours, sizeof(CvContour), retr, method, CvPoint(offset));

    if (!_ccontours)
    {
        contours.clear();
        return;
    }
    Seq<CvSeq*> all_contours(cvTreeToNodeSeq(_ccontours, sizeof(CvSeq), storage));
    int total = (int)all_contours.size();
    contours.resize(total);

    SeqIterator<CvSeq*> it = all_contours.begin();
    for (int i = 0; i < total; i++, ++it)
    {
        CvSeq* c = *it;
        ((CvContour*)c)->color = (int)i;
        int count = (int)c->total;
        int* data = new int[count * 2];
        cvCvtSeqToArray(c, data);
        for (int j = 0; j < count; j++)
        {
            contours[i].push_back(Point(data[j * 2], data[j * 2 + 1]));
        }
        delete[] data;
    }

    hierarchy.resize(total);
    it = all_contours.begin();
    for (int i = 0; i < total; i++, ++it)
    {
        CvSeq* c = *it;
        int h_next = c->h_next ? ((CvContour*)c->h_next)->color : -1;
        int h_prev = c->h_prev ? ((CvContour*)c->h_prev)->color : -1;
        int v_next = c->v_next ? ((CvContour*)c->v_next)->color : -1;
        int v_prev = c->v_prev ? ((CvContour*)c->v_prev)->color : -1;
        hierarchy[i] = Vec4i(h_next, h_prev, v_next, v_prev);
    }
    storage.release();
}


//一张图中红细胞的参数
struct RedCellInfoPerImg
{
    int redCellCounts;   //红细胞数量
    double avgR;   //红细胞平均半径
    int twoAdhCount;   //2黏连红细胞数量
    int threeAdhCount;   //3黏连红细胞数量
    int overThreeAdhCount;   //大于3黏连红细胞数量
    int overThreeAdhSimCellCount;//大于3黏连红细胞切分后红细胞数量
    int lightDiff02;  //背景平均亮度 - 红细胞区的平均亮度  用double是因为我觉得很有可能和后面会改成比值
    int lightDiff03;
    int lightDiff04;
    int lightDiff05;
    double redCellDistanceAvg;
    RedCellInfoPerImg() :redCellCounts(0), avgR(0), twoAdhCount(0),
        threeAdhCount(0), overThreeAdhCount(0), overThreeAdhSimCellCount(0),
        lightDiff02(0), lightDiff03(0), lightDiff04(0), lightDiff05(0), redCellDistanceAvg(0)
    {}
};

void ResizeImage(Mat& srcImage)
{
    int width = 1224;
    int height = 1024;
    resize(srcImage, srcImage, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
}

vector<coord_t> GetRedCellsCount(Mat rgbMat, RedCellInfoPerImg& redCellInfo)
{
    ResizeImage(rgbMat);
    Mat imgGray(rgbMat.rows, rgbMat.cols, CV_8UC1);
    cvtColor(rgbMat, imgGray, CV_RGB2GRAY);
    vector<coord_t> current_coord_vec;

    //二值化
    Mat mask = Mat(imgGray.rows, imgGray.cols, CV_8UC1, Scalar(0));
    //cv::threshold(imgGray, mask, 0, 255, THRESH_OTSU); 
    cv::threshold(imgGray, mask, 0, 255, THRESH_TRIANGLE);
    mask = 255 - mask;

    vector<Vec4i> hierarchy;
    std::vector<std::vector<cv::Point>> contours;

    //cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE 检测最外层轮廓
    findContours2(mask.clone(), contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    int count = contours.size();

    double redCellArea = 0;  //未减少时候的面积和
    for (int i = 0; i < count; ++i)
    {
        Rect rect = cv::boundingRect(contours[i]);
        if (rect.width > 5 && rect.height > 5)
        {
            redCellArea += contourArea(contours[i]);
        }
    }

    //单个红细胞的灰度值
    redCellInfo.redCellCounts = 0;
    int redCellGrayAvg = 170;
    float saveRatio = 0.9;
    float R = 2 * r;  //直径

    redCellInfo.avgR = 0;  //平均直径

                           //存储单个细胞的中心点位置
    vector<Point2i> singleCellCenterPoint;


    for (int i = 0; i < count; ++i)
    {
        Rect rect = cv::boundingRect(contours[i]);

        if (rect.width < R*0.9 || rect.height < R*0.9) {
            continue;
        }

        cv::rectangle(rgbMat, rect, Scalar(255, 0, 0), 1, LINE_8, 0);
        if (rect.width <= (2 * saveRatio * R) && rect.height <= (2 * saveRatio * R))
        {


            ++redCellInfo.redCellCounts;
            redCellInfo.avgR += (rect.width + rect.height);
            singleCellCenterPoint.push_back(Point2i(rect.x + rect.width / 2, rect.y + rect.height / 2));
            cv::putText(rgbMat, "1", Point2i(rect.x + rect.width / 2, rect.y + rect.height / 2), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 255), 2, 2, 0);
            continue;
        }
        if (rect.width <= (3 * saveRatio * R) && rect.height <= (3 * saveRatio * R))
        {
            redCellInfo.redCellCounts += 2;
            redCellInfo.avgR += (rect.width + rect.height);
            ++redCellInfo.twoAdhCount;
            cv::putText(rgbMat, "2", Point2i(rect.x + rect.width / 2, rect.y + rect.height / 2), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 0, 255), 2, 2, 0);
            continue;
        }
        if (rect.width <= (4 * saveRatio * R) && rect.height <= (4 * saveRatio * R))
        {
            redCellInfo.redCellCounts += 3;
            redCellInfo.avgR += (rect.width + rect.height);
            ++redCellInfo.threeAdhCount;
            cv::putText(rgbMat, "3", Point2i(rect.x + rect.width / 2, rect.y + rect.height / 2), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 0), 2, 2, 0);
            continue;
        }
        ++redCellInfo.overThreeAdhCount;
        double cur_area = contourArea(contours[i]);
        //重叠多的用 近似数量 然后*1.2
        float overlay_rate = 1.5;
        int simCount = (cur_area * overlay_rate) / (PI * 1.5 * r * r);
        //cv::putText(rgbMat, to_string(simCount), Point2i(rect.x + rect.width / 2, rect.y + rect.height / 2), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 0), 2, 2, 0);

        redCellInfo.overThreeAdhSimCellCount += simCount;
    }

    //计算均值
    int singleCellCount = singleCellCenterPoint.size();

    if (singleCellCount > 50)
    {
        vector<double> min_distance_vtr;
        min_distance_vtr.resize(singleCellCount);
        double sum = 0;
        for (int i = 0; i < singleCellCount; ++i)
        {
            double min_distance = numeric_limits<double>::max();
            for (int j = 0; j < singleCellCount; ++j)
            {
                if (i == j) continue;
                double cur_distance = powf((singleCellCenterPoint[i].x - singleCellCenterPoint[j].x), 2)
                    + powf((singleCellCenterPoint[i].y - singleCellCenterPoint[j].y), 2);
                cur_distance = sqrtf(cur_distance);
                if (min_distance > cur_distance)
                {
                    min_distance = cur_distance;
                    min_distance_vtr[i] = min_distance;
                }
            }
            sum += min_distance_vtr[i];
        }

        double avg_distance = sum / min_distance_vtr.size(); //均值  
        redCellInfo.redCellDistanceAvg = avg_distance;
    }
    else
    {
        redCellInfo.redCellDistanceAvg = 0;
    }
    cout << redCellInfo.redCellDistanceAvg << endl;
    //获取小于阈值的区域
    float ratio = 0.1;
    int threhold = (int)(redCellGrayAvg * (1 - ratio));

    //获取新的mask0  1是细胞区域
    Mat mask0 = mask / 255;
    Mat result = imgGray.mul(mask0);
    cv::threshold(result, mask0, threhold, 255, THRESH_BINARY);

    //计算mask差值
    mask -= mask0;
    mask /= 255;
    redCellArea -= sum(mask)[0];
    if (redCellInfo.redCellCounts != 0)
        redCellInfo.avgR /= redCellInfo.redCellCounts * 2;
    else
        redCellInfo.avgR = 0;

    //imshow("dd", rgbMat);
    //waitKey();
    //destroyAllWindows();

    return current_coord_vec;
}



#pragma endregion



double GetAverageAreaOfCells(string folder)
{
    //string folder = "F:\\Workdatas\\OnelayerCells\\hhh\\1.2-1-1";
    string filter = "*.jpg";

    double sumAll = 0;

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));//创建结构元素大小为3*3

    FindFiles ff;
    auto images = ff.findFiles(folder, filter);
    auto imgCount = images.size();
    int i = 0;
    for each (auto imgpath in images)
    {
        //cout << imgpath << endl;
        
        Mat img = imread(imgpath, 0);
        Mat cellArea;
        //threshold(img, cellArea, 0, 255, THRESH_OTSU);
        threshold(img, cellArea, 178, 255, THRESH_BINARY_INV);
        morphologyEx(cellArea, cellArea, MORPH_OPEN, kernel);
        auto v = sum(cellArea)[0] / 255.0 / 1000;
        

        /*
        Mat img = imread(imgpath, 1);
        RedCellInfoPerImg redCellInfo;
        GetRedCellsCount(img, redCellInfo);
        auto v = redCellInfo.redCellCounts;
        */

        sumAll += v;
        cout << imgCount << " \\ " << i++ << "  " << v << endl;
       /* cout << v << endl;
        imshow("eee", img);
        imshow("ddd", cellArea);
        waitKey();
        destroyAllWindows();*/
        
    }

    sumAll /= images.size();
  
    return sumAll;
}

int main() 
{
    /*string folders[] = {
        "F:\\Workdatas\\OnelayerCells\\hhh\\1.2-1-1",
        "F:\\Workdatas\\OnelayerCells\\hhh\\1.2-1-2",
        "F:\\Workdatas\\OnelayerCells\\hhh\\1.2-2-1",
        "F:\\Workdatas\\OnelayerCells\\hhh\\1.2-2-2",
        "F:\\Workdatas\\OnelayerCells\\hhh\\1.2-2-3",
        "F:\\Workdatas\\OnelayerCells\\hhh\\1.5-1-1",
        "F:\\Workdatas\\OnelayerCells\\hhh\\1.5-1-2",
        "F:\\Workdatas\\OnelayerCells\\hhh\\1.5-1-3",
        "F:\\Workdatas\\OnelayerCells\\hhh\\1.5-2-1",
        "F:\\Workdatas\\OnelayerCells\\hhh\\1.5-2-2"
    };
*/
    string folders[] = {
        "J:\\Gray\\hhh2\\1-10ul",
        "J:\\Gray\\hhh2\\2-14ul",
        "J:\\Gray\\hhh2\\3-14pul",
        "J:\\Gray\\hhh2\\4-14ppul",
        "J:\\Gray\\hhh2\\5-14ul",
    };

    vector<double> averageAreas;

    for (int i = 0; i < sizeof(folders) / sizeof(string); i++)
    {
        string folder = folders[i];
        double v = GetAverageAreaOfCells(folder);
        averageAreas.push_back(v);
    }

    cout << "======================================" << endl;
    for (int i = 0; i < averageAreas.size(); i++) {
        cout << folders[i] << "    : " << averageAreas[i] << endl;
    }


    cin.ignore();
    return 0;
}