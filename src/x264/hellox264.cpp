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

//* ���ò���   
void EncoderInit(x264_param_t* pX264Param, int width, int height)
{
	//* ʹ��Ĭ�ϲ�������������Ϊ�ҵ���ʵʱ���紫�䣬������ʹ����zerolatency��ѡ�ʹ�����ѡ��֮��Ͳ�����delayed_frames�������ʹ�õĲ��������Ļ�������Ҫ�ڱ������֮��õ�����ı���֡   
	x264_param_default_preset(pX264Param, "veryfast", "zerolatency");

	//* cpuFlags   
	pX264Param->i_threads = X264_SYNC_LOOKAHEAD_AUTO;//* ȡ�ջ���������ʹ�ò������ı�֤.   
													 //* ��Ƶѡ��      
	pX264Param->i_width = width; //* Ҫ�����ͼ����.   
	pX264Param->i_height = height; //* Ҫ�����ͼ��߶�   
	pX264Param->i_frame_total = 0; //* ������֡��.��֪����0.   
	pX264Param->i_keyint_max = 10;

	//* ������   
	pX264Param->i_bframe = 5;
	pX264Param->b_open_gop = 0;
	pX264Param->i_bframe_pyramid = 0;
	pX264Param->i_bframe_adaptive = X264_B_ADAPT_TRELLIS;

	//* Log����������Ҫ��ӡ������Ϣʱֱ��ע�͵�����   
	pX264Param->i_log_level = X264_LOG_DEBUG;

	//* ���ʿ��Ʋ���   
	pX264Param->rc.i_bitrate = 1024 * 10;//* ����(������,��λKbps)  

										 //* muxing parameters   
	pX264Param->i_fps_den = 1; //* ֡�ʷ�ĸ   
	pX264Param->i_fps_num = 10;//* ֡�ʷ���   
	pX264Param->i_timebase_den = pX264Param->i_fps_num;
	pX264Param->i_timebase_num = pX264Param->i_fps_den;

	//* ����Profile.ʹ��Baseline profile   
	x264_param_apply_profile(pX264Param, x264_profile_names[0]);
	return;
}


x264_t* handleGet(int width, int height)
{
	x264_t* pX264Handle = NULL;

	/*videoShow avi;*/
	x264_param_t* pX264Param = new x264_param_t;
	assert(pX264Param);

	//* ���ò���   
	EncoderInit(pX264Param, width, height);

	//* �򿪱��������,ͨ��x264_encoder_parameters�õ����ø�X264�Ĳ���.ͨ��x264_encoder_reconfig����X264�Ĳ���   
	pX264Handle = x264_encoder_open(pX264Param);
	assert(pX264Handle);

	delete pX264Param;
	pX264Param = NULL;

	return pX264Handle;
}

x264_picture_t * picInInit(int width, int height)
{
	x264_picture_t* pPicIn = NULL;

	//* ������Ҫ�ĸ�������   
	pPicIn = new x264_picture_t;
	x264_picture_alloc(pPicIn, X264_CSP_I420, width, height);
	pPicIn->img.i_csp = X264_CSP_I420;  /* yuv 4:2:0 planar */
	pPicIn->img.i_plane = 3;            /* Number of image planes: 3 ��ͼ��ƽ�� Y,U,V */

	return pPicIn;
}


void imgPalneCopy(x264_picture_t* pPicIn, Mat yuv)
{
	//��YUV�������ݷֱ𿽱�������ƽ�棬����plane[i]����ֿ����YUV��������

	vector<Mat> yuvarr;
	cv::split(yuv, yuvarr);
	memcpy(pPicIn->img.plane[0], yuvarr[0].datastart, yuv.cols*yuv.rows);
	memcpy(pPicIn->img.plane[0], yuvarr[1].datastart, yuv.cols*yuv.rows/4);
	memcpy(pPicIn->img.plane[0], yuvarr[2].datastart, yuv.cols*yuv.rows/4);
}

void imgPalneCopy2(x264_picture_t* pPicIn, Mat bgr)
{
	//��YUV�������ݷֱ𿽱�������ƽ�棬����plane[i]����ֿ����BGR��������

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

	//YUV3��ͨ��������ͼƬ��
	imgPalneCopy(pPicIn, yuv);


	pPicIn->i_pts = frameNum;
	
	//* ������Ҫ�ĸ�������   
	int iNal = 0;
	x264_nal_t *pNals = NULL;
	x264_picture_t* pPicOut = new x264_picture_t;
	x264_picture_init(pPicOut);

	//����һ֡   
	int frame_size = x264_encoder_encode(pX264Handle, &pNals, &iNal, pPicIn, pPicOut);

	if (frame_size >0)
	{
		for (int i = 0; i < iNal; ++i)
		{
			//����������д���ļ�.   
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