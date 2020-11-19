#include "libs.h"

#include <iostream>
#include <string>
#include <stdint.h>
#include <stdio.h>

extern "C"
{
#include "x264.h"
}

#pragma comment(lib, "libx264.lib")

using namespace std;
using namespace cv;


int dispCamera() 
{
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		return -1;
	}
	Mat frame;
	bool stop = false;
	while (!stop) {
		cap >> frame;
		imshow("camera", frame);
		if (waitKey(30) >= 0) {
			stop = true;
		}
	}

	return 0;
}

//* 配置参数   
void EncoderInit(x264_param_t* pX264Param, int width, int height)
{
	//* 使用默认参数，在这里因为我的是实时网络传输，所以我使用了zerolatency的选项，使用这个选项之后就不会有delayed_frames，如果你使用的不是这样的话，还需要在编码完成之后得到缓存的编码帧   
	x264_param_default_preset(pX264Param, "veryfast", "zerolatency");

	//* cpuFlags   
	pX264Param->i_threads = X264_SYNC_LOOKAHEAD_AUTO;//* 取空缓冲区继续使用不死锁的保证.   
													 //* 视频选项      
	pX264Param->i_width = width; //* 要编码的图像宽度.   
	pX264Param->i_height = height; //* 要编码的图像高度   
	pX264Param->i_frame_total = 0; //* 编码总帧数.不知道用0.   
	pX264Param->i_keyint_max = 10;

	//* 流参数   
	pX264Param->i_bframe = 5;
	pX264Param->b_open_gop = 0;
	pX264Param->i_bframe_pyramid = 0;
	pX264Param->i_bframe_adaptive = X264_B_ADAPT_TRELLIS;

	//* Log参数，不需要打印编码信息时直接注释掉就行   
	pX264Param->i_log_level = X264_LOG_DEBUG;

	//* 速率控制参数   
	pX264Param->rc.i_bitrate = 1024 * 10;//* 码率(比特率,单位Kbps)  

										 //* muxing parameters   
	pX264Param->i_fps_den = 1; //* 帧率分母   
	pX264Param->i_fps_num = 10;//* 帧率分子   
	pX264Param->i_timebase_den = pX264Param->i_fps_num;
	pX264Param->i_timebase_num = pX264Param->i_fps_den;

	//* 设置Profile.使用Baseline profile   
	x264_param_apply_profile(pX264Param, x264_profile_names[0]);
	return;
}


x264_t* handleGet(int width, int height)
{
	x264_t* pX264Handle = NULL;

	/*videoShow avi;*/
	x264_param_t* pX264Param = new x264_param_t;
	assert(pX264Param);

	//* 配置参数   
	EncoderInit(pX264Param, width, height);

	//* 打开编码器句柄,通过x264_encoder_parameters得到设置给X264的参数.通过x264_encoder_reconfig更新X264的参数   
	pX264Handle = x264_encoder_open(pX264Param);
	assert(pX264Handle);

	delete pX264Param;
	pX264Param = NULL;

	return pX264Handle;
}

x264_picture_t * picInInit(int width, int height)
{
	x264_picture_t* pPicIn = NULL;

	//* 编码需要的辅助变量   
	pPicIn = new x264_picture_t;
	x264_picture_alloc(pPicIn, X264_CSP_I420, width, height);
	pPicIn->img.i_csp = X264_CSP_I420;  /* yuv 4:2:0 planar */
	pPicIn->img.i_plane = 3;            /* Number of image planes: 3 个图像平面 Y,U,V */

	return pPicIn;
}


void imgPalneCopy(x264_picture_t* pPicIn, Mat yuv)
{
	//把YUV分量数据分别拷贝到三个平面，三个plane[i]数组分开存放YUV三个分量

	vector<Mat> yuvarr;
	cv::split(yuv, yuvarr);
	memcpy(pPicIn->img.plane[0], yuvarr[0].datastart, yuv.cols*yuv.rows);
	memcpy(pPicIn->img.plane[0], yuvarr[1].datastart, yuv.cols*yuv.rows/4);
	memcpy(pPicIn->img.plane[0], yuvarr[2].datastart, yuv.cols*yuv.rows/4);
}

void imgPalneCopy2(x264_picture_t* pPicIn, Mat bgr)
{
	//把YUV分量数据分别拷贝到三个平面，三个plane[i]数组分开存放BGR三个分量

	vector<Mat> bgrarr;
	cv::split(bgr, bgrarr);
	memcpy(pPicIn->img.plane[0], bgrarr[0].datastart, bgr.cols*bgr.rows);
	memcpy(pPicIn->img.plane[0], bgrarr[1].datastart, bgr.cols*bgr.rows);
	memcpy(pPicIn->img.plane[0], bgrarr[2].datastart, bgr.cols*bgr.rows);
}

void encodeOneFrame(x264_t* pX264Handle, x264_picture_t* pPicIn, Mat frame, int64 frameNum )
{
	// bgr2yuv400
	Mat yuv;
	cvtColor(frame, yuv, CV_BGR2YUV_I420);

	//YUV3个通道拷贝到图片中
	imgPalneCopy(pPicIn, yuv);


	pPicIn->i_pts = frameNum;
	
	//* 编码需要的辅助变量   
	int iNal = 0;
	x264_nal_t *pNals = NULL;
	x264_picture_t* pPicOut = new x264_picture_t;
	x264_picture_init(pPicOut);

	//编码一帧   
	int frame_size = x264_encoder_encode(pX264Handle, &pNals, &iNal, pPicIn, pPicOut);

	if (frame_size >0)
	{
		for (int i = 0; i < iNal; ++i)
		{
			//将编码数据写入文件.   
			//fwrite(pNals[i].p_payload, 1, pNals[i].i_payload, pFile);
		}
	}

	delete pPicOut;
	pPicOut = NULL;
}

int CameraToX264(x264_t* pX264Handle, x264_picture_t* pPicIn)
{
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		return -1;
	}
	Mat frame;
	int frameId = 0;
	bool stop = false;
	while (!stop) {
		cap >> frame;
		imshow("camera", frame);

		encodeOneFrame(pX264Handle, pPicIn, frame, frameId++);
		
		if (waitKey(30) >= 0) {
			stop = true;
		}
	}

	return 0;
}

int hellox264()
{
	const int width = 640;
	const int height = 480;
	x264_t* pX264Handle = handleGet(width, height);
	x264_picture_t* pPicIn = picInInit(width, height);

	CameraToX264(pX264Handle, pPicIn);

	return 0;
}

int main()
{
	// return dispCamera();
	return hellox264();
}