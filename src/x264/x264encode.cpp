
#include "StdAfx.h"
#include "stdint.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
extern "C"
{  
	#include"x264.h"
}  

#pragma comment(lib, "libx264.lib       ")
#pragma comment(lib, "opencv_world342d.lib")



//ת������
#define MY(a,b,c) (( a*  0.2989  + b*  0.5866  + c*  0.1145))
#define MU(a,b,c) (( a*(-0.1688) + b*(-0.3312) + c*  0.5000 + 128))
#define MV(a,b,c) (( a*  0.5000  + b*(-0.4184) + c*(-0.0816) + 128))

//��С�ж�
#define DY(a,b,c) (MY(a,b,c) > 255 ? 255 : (MY(a,b,c) < 0 ? 0 : MY(a,b,c)))
#define DU(a,b,c) (MU(a,b,c) > 255 ? 255 : (MU(a,b,c) < 0 ? 0 : MU(a,b,c)))
#define DV(a,b,c) (MV(a,b,c) > 255 ? 255 : (MV(a,b,c) < 0 ? 0 : MV(a,b,c)))
#define CLIP(a) ((a) > 255 ? 255 : ((a) < 0 ? 0 : (a)))





class videoShow
{
public:
	CvVideoWriter* writer;

	videoShow();
	~videoShow();
	IplImage * currentFrameGet();
	int currentFrameNum();
	bool aviSave(IplImage* p);
	bool frameSave(IplImage * pFrame,char * frameOutName);
	void videoOpen(char* videoName);
	void parseCmdArgs(int argc, char** argv);
	CvSize sizeGet();
	double fpsGet();

private:
	double fps;
	CvSize size;
	int pFrameNum;
	CvCapture* pCapture;
};

CvSize videoShow :: sizeGet()
{
	if(NULL == pCapture)
	{
		size.height = 0;
		size.width = 0;
		return size;
	}

	 size = cvSize(
		(int)cvGetCaptureProperty( pCapture, CV_CAP_PROP_FRAME_WIDTH),
		(int)cvGetCaptureProperty( pCapture, CV_CAP_PROP_FRAME_HEIGHT)
		);
	//printf("frame (w, h) = (%d, %d)\n",size.width,size.height);
	
	return size;
}

double videoShow :: fpsGet()
{
	if(NULL == pCapture)
	{
		return -1;
	}
	
	 fps = cvGetCaptureProperty (
		pCapture,
		CV_CAP_PROP_FPS
		);

	//printf("fps=%d\n",(int)fps);
	
	return fps;
}

videoShow :: videoShow()
{
	pFrameNum = 0;
	writer = NULL;
	pCapture = NULL;
	
	//Ĭ��Ϊ��ȡ����ͷ��Ƶ
	if( !(pCapture = cvCaptureFromCAM(0)))
	{
		size.height = 0;
		size.width = 0;
		fps = 0;

		fprintf(stderr, "Can not open camera.\n");
		return;
	}

	fps = cvGetCaptureProperty (
		pCapture,
		CV_CAP_PROP_FPS
		);

	//printf("fps=%d\n",(int)fps);

	size = cvSize(
		(int)cvGetCaptureProperty( pCapture, CV_CAP_PROP_FRAME_WIDTH),
		(int)cvGetCaptureProperty( pCapture, CV_CAP_PROP_FRAME_HEIGHT)
		);
	//printf("frame (w, h) = (%d, %d)\n",size.width,size.height);

	return;
}

videoShow :: ~videoShow()
{
	if(NULL != pCapture)
	{
		cvReleaseCapture(&pCapture);
	}
	if(NULL != writer)
	{
		cvReleaseVideoWriter( &writer );
	}
}

//���������ͷ����ȡ���ʱ�̵�֡���������Ƶ����ȡһ֡�����ص�pFrameOutʹ����֮��Ҫ�ͷţ�������ڴ�й©
IplImage * videoShow :: currentFrameGet()
{
	if(NULL == pCapture)
	{
		fprintf(stderr, "Can not get currentFrame.\n");
		return NULL;
	}

	IplImage* pFrame = NULL; 
	IplImage* pFrameOut = NULL;

	//����cvQueryFrame������ͷ�����ļ���ץȡһ֡��Ȼ���ѹ��������һ֡��������������Ǻ���cvGrabFrame�ͺ���cvRetrieveFrame��һ����õ���ϡ����ص�ͼ�񲻿��Ա��û��ͷŻ����޸�
	pFrame = cvQueryFrame( pCapture );

	pFrameOut = cvCreateImage(cvSize(pFrame->width,pFrame->height),pFrame->depth, pFrame->nChannels);
	if (NULL == pFrameOut)
	{
		fprintf(stderr, "Can not create frame.\n");
		return NULL;
	}

	//cvCloneImage�����������������֤����ֽϴ���ڴ�й¶������Ȼ�����ͷţ���������Ӳ�֪���������ͷţ���Ϊ��ÿ�ο���������ͼ���������������ͷ��ROI�����ݡ�ÿ��ʹ��ʱ������������µ��ڴ�ռ䣬���Ḳ����ǰ�����ݡ�һ��752*480��С������С��ͼ��ÿ��й¶���ڴ��ԼΪ1M��
	//���������ʹ��cvCopy��������
	cvCopy(pFrame,pFrameOut,NULL);  
	pFrameNum++;

	return pFrameOut;
}

//��ȡ��ǰ֡��
int videoShow :: currentFrameNum()
{
	return pFrameNum;
}

//�ѵ�ǰ֡���浽avi��Ƶ��
bool videoShow :: aviSave(IplImage* p)
{
	if ((NULL == writer) || (NULL == p))
	{
		return false;
	}
	cvWriteToAVI(writer, p);
	return true;
}

//����ͼƬ�����û������ͼƬ������Ĭ�ϱ���ΪresultPic.jpg
bool videoShow :: frameSave(IplImage * pFrame,char * frameOutName)
{
	if (NULL == pFrame)
	{
		return false;
	}

	if (NULL == frameOutName)
	{
		cvSaveImage("resultPic.jpg",pFrame);
		return true;
	}

	cvSaveImage(frameOutName,pFrame);
	return true;
}

//��ȡargc��argv����
void videoShow :: parseCmdArgs(int argc, char** argv)
{
	if(NULL != pCapture)
	{
		cvReleaseCapture(&pCapture);
		CvCapture* pCapture = NULL;
	}
	
	if (argc ==1)
	{
		if( !(pCapture = cvCaptureFromCAM(0)))
		{
			fprintf(stderr, "Can not open camera.\n");
			return;
		}
	}

	//����Ƶ�ļ�
	if(argc == 2)
	{
		if( !(pCapture = cvCaptureFromFile(argv[1])))
		{
			fprintf(stderr, "Can not open video file %s\n", argv[1]);
			return;
		}
	}

	return;
}

//��ȡ��Ƶ
void videoShow :: videoOpen(char* videoName)
{
	if(NULL != pCapture)
	{
		cvReleaseCapture(&pCapture);
		CvCapture* pCapture = NULL;
	}
	
	//����Ƶ�ļ�
	if( !(pCapture = cvCaptureFromFile(videoName)))
	{
		fprintf(stderr, "Can not open video file %s\n", videoName);
		return;
	}

	fps = cvGetCaptureProperty (
		pCapture,
		CV_CAP_PROP_FPS
		);

	//printf("fps=%d\n",(int)fps);

	size = cvSize(
		(int)cvGetCaptureProperty( pCapture, CV_CAP_PROP_FRAME_WIDTH),
		(int)cvGetCaptureProperty( pCapture, CV_CAP_PROP_FRAME_HEIGHT)
		);
	//printf("frame (w, h) = (%d, %d)\n",size.width,size.height);

	return;
}

class enCODE264
{
public:
	enCODE264(IplImage * pFrame,FILE * pF = NULL);
	~enCODE264();
	void encodeOneFrame(IplImage * pFrame);
	uint8_t * encodeOneFrame2(IplImage * pFrame, float * length, bool type = true);
	int getFrameNum();
	void reSet264(IplImage * pFrame,FILE * pF = NULL);
	FILE * getPFile();

private:
	int frameNum;
	x264_t* pX264Handle; 
	x264_picture_t* pPicIn;
	FILE * pFile;

	x264_picture_t * picInInit(IplImage * pFrame);
	x264_t* handleGet(IplImage * pFrame);
	void EncoderInit(x264_param_t* pX264Param, int width, int height);

	void cvBRGtoRGB(unsigned char *RGB, IplImage * pFrame);
	void Convert(unsigned char *RGB, unsigned char *YUV,unsigned int width,unsigned int height);
	void imgPalneCopy(x264_picture_t* pPicIn, unsigned char *YUV1, IplImage* pFrame);

	void framecodeWrite(x264_t* pX264Handle, x264_picture_t* pPicIn,FILE * pFile);
	uint8_t * framecodeSend(x264_t* pX264Handle, x264_picture_t* pPicIn,float * length);
};

enCODE264 :: enCODE264(IplImage * pFrame,FILE * pF)
{
	frameNum = 0;

	//��ʼ�����
	pX264Handle = handleGet(pFrame);

	//��ʼ��
	pPicIn = picInInit(pFrame);

	if (NULL == pF)
	{
		//* ����Ĭ���ļ�,���ڴ洢��������   
		pFile = fopen("opencv.h264", "wb"); 
		assert(pFile); 
		return;
	}
	pFile = pF;
}

FILE * enCODE264 :: getPFile()
{
	return pFile;
}

void enCODE264 :: reSet264(IplImage * pFrame,FILE * pF)
{
	//* ���ͼ������   
	if (NULL != pPicIn)
	{
		x264_picture_clean(pPicIn);  
		delete pPicIn;  
		pPicIn = NULL; 
	}

	if (NULL != pX264Handle)
	{
		//* �رձ��������   
		x264_encoder_close(pX264Handle);  
		pX264Handle = NULL;  
	}

	frameNum = 0;

	//��ʼ�����
	pX264Handle = handleGet(pFrame);

	//��ʼ��
	pPicIn = picInInit(pFrame);

	if (NULL == pF)
	{
		//* ����Ĭ���ļ�,���ڴ洢��������   
		pFile = fopen("opencv.h264", "wb"); 
		assert(pFile); 
		return;
	}
	pFile = pF;

}

enCODE264 :: ~enCODE264()
{
	//* ���ͼ������   
	if (NULL != pPicIn)
	{
		x264_picture_clean(pPicIn);  
		delete pPicIn;  
		pPicIn = NULL; 
	}

	if (NULL != pX264Handle)
	{
		//* �رձ��������   
		x264_encoder_close(pX264Handle);  
		pX264Handle = NULL;  
	}
}

int enCODE264 :: getFrameNum()
{
	return frameNum;
}

//* ���ò���   
void enCODE264 :: EncoderInit(x264_param_t* pX264Param, int width, int height)
{
	//* ʹ��Ĭ�ϲ�������������Ϊ�ҵ���ʵʱ���紫�䣬������ʹ����zerolatency��ѡ�ʹ�����ѡ��֮��Ͳ�����delayed_frames�������ʹ�õĲ��������Ļ�������Ҫ�ڱ������֮��õ�����ı���֡   
	x264_param_default_preset(pX264Param, "veryfast", "zerolatency");  

	//* cpuFlags   
	pX264Param->i_threads  = X264_SYNC_LOOKAHEAD_AUTO;//* ȡ�ջ���������ʹ�ò������ı�֤.   
	//* ��Ƶѡ��      
	pX264Param->i_width   = width; //* Ҫ�����ͼ����.   
	pX264Param->i_height  = height; //* Ҫ�����ͼ��߶�   
	pX264Param->i_frame_total = 0; //* ������֡��.��֪����0.   
	pX264Param->i_keyint_max = 10;   

	//* ������   
	pX264Param->i_bframe  = 5;  
	pX264Param->b_open_gop  = 0;  
	pX264Param->i_bframe_pyramid = 0;  
	pX264Param->i_bframe_adaptive = X264_B_ADAPT_TRELLIS; 

	//* Log����������Ҫ��ӡ������Ϣʱֱ��ע�͵�����   
	pX264Param->i_log_level  = X264_LOG_DEBUG;  

	//* ���ʿ��Ʋ���   
	pX264Param->rc.i_bitrate = 1024 * 10;//* ����(������,��λKbps)  

	//* muxing parameters   
	pX264Param->i_fps_den  = 1; //* ֡�ʷ�ĸ   
	pX264Param->i_fps_num  = 10;//* ֡�ʷ���   
	pX264Param->i_timebase_den = pX264Param->i_fps_num;  
	pX264Param->i_timebase_num = pX264Param->i_fps_den;  

	//* ����Profile.ʹ��Baseline profile   
	x264_param_apply_profile(pX264Param, x264_profile_names[0]); 
	return;
}

/*-----------------����ɫ�ȿռ����YUV420�����ڴ棬�������ڴ���׵�ַ��Ϊָ��---------------
���ǳ�˵��YUV420����planar��ʽ��YUV��ʹ����������ֿ����YUV����������������
һ����άƽ��һ�����ڳ���H264���Ե�YUV������,����CIFͼ���С��YUV����(352*288),��
�ļ���ʼ��û���ļ�ͷ,ֱ�Ӿ���YUV����,�ȴ��һ֡��Y��Ϣ,����Ϊ352*288��byte, Ȼ���ǵ�
һ֡U��Ϣ������352*288/4��byte, ����ǵ�һ֡��V��Ϣ,������352*288/4��byte, ��˿�����
����һ֡�����ܳ�����352*288*1.5,��152064��byte, ������������300֡�Ļ�, ��ô������
���ȼ�Ϊ152064*300=44550KB,��Ҳ����Ϊʲô������300֡CIF��������44M��ԭ��.
---------------------------------------------------------------------------------*/
void enCODE264 :: Convert(unsigned char *RGB, unsigned char *YUV,unsigned int width,unsigned int height)
{
	//��������
	unsigned int i,x,y,j;
	unsigned char *Y = NULL;
	unsigned char *U = NULL;
	unsigned char *V = NULL;

	Y = YUV;
	U = YUV + width*height;
	V = U + ((width*height)>>2);
	for(y=0; y < height; y++)
		for(x=0; x < width; x++)
		{
			j = y*width + x;
			i = j*3;
			Y[j] = (unsigned char)(DY(RGB[i], RGB[i+1], RGB[i+2]));
			if(x%2 == 1 && y%2 == 1)
			{
				j = (width>>1) * (y>>1) + (x>>1);
				//����i����Ч
				U[j] = (unsigned char)
					((DU(RGB[i  ], RGB[i+1], RGB[i+2]) + 
					DU(RGB[i-3], RGB[i-2], RGB[i-1]) +
					DU(RGB[i  -width*3], RGB[i+1-width*3], RGB[i+2-width*3]) +
					DU(RGB[i-3-width*3], RGB[i-2-width*3], RGB[i-1-width*3]))/4);
				V[j] = (unsigned char)
					((DV(RGB[i  ], RGB[i+1], RGB[i+2]) + 
					DV(RGB[i-3], RGB[i-2], RGB[i-1]) +
					DV(RGB[i  -width*3], RGB[i+1-width*3], RGB[i+2-width*3]) +
					DV(RGB[i-3-width*3], RGB[i-2-width*3], RGB[i-1-width*3]))/4);
			}
		}
}

void enCODE264 :: imgPalneCopy(x264_picture_t* pPicIn, unsigned char *YUV1, IplImage* pFrame)
{
		//��YUV�������ݷֱ𿽱�������ƽ�棬����plane[i]����ֿ����YUV��������
		unsigned char *p1,*p2;
		p1=YUV1;
		p2=pPicIn->img.plane[0];
		for(int i=0;i<pFrame->height;i++)
		{
			memcpy(p2,p1,pFrame->width);
			p1+=pFrame->width;
			p2+=pFrame->width;
		}

		p2=pPicIn->img.plane[1];
		for(int i=0;i<pFrame->height/2;i++)
		{
			memcpy(p2,p1,pFrame->width/2);
			p1+=pFrame->width/2;
			p2+=pFrame->width/2;
		}

		p2=pPicIn->img.plane[2];
		for(int i=0;i<pFrame->height/2;i++)
		{
			memcpy(p2,p1,pFrame->width/2);
			p1+=pFrame->width/2;
			p2+=pFrame->width/2;
		}
}

void enCODE264 :: cvBRGtoRGB(unsigned char *RGB, IplImage * pFrame)
{
	//pFrame��BGRתRGB
	for(int i=0;i<pFrame->height;i++)
	{
		for(int j=0;j<pFrame->width;j++)
		{
			//��opencv��pFrame��ȡλͼRGB����,����BGRתRGB
			RGB[(i*pFrame->width+j)*3]   = pFrame->imageData[i * pFrame->widthStep + j * 3 + 2];
			RGB[(i*pFrame->width+j)*3+1] = pFrame->imageData[i * pFrame->widthStep + j * 3 + 1];
			RGB[(i*pFrame->width+j)*3+2] = pFrame->imageData[i * pFrame->widthStep + j * 3  ]; 
		}
	}
}

void enCODE264 :: framecodeWrite(x264_t* pX264Handle, x264_picture_t* pPicIn,FILE * pFile)
{
	//* ������Ҫ�ĸ�������   
	int iNal = 0;  
	x264_nal_t *pNals = NULL;  
	x264_picture_t* pPicOut = new x264_picture_t;  
	x264_picture_init(pPicOut);  

	//����һ֡   
	int frame_size = x264_encoder_encode(pX264Handle,&pNals,&iNal,pPicIn,pPicOut);  

	if(frame_size >0)  
	{  
		for (int i = 0; i < iNal; ++i)  
		{
			//����������д���ļ�.   
			fwrite(pNals[i].p_payload, 1, pNals[i].i_payload, pFile);  
		}  
	}  

	delete pPicOut;  
	pPicOut = NULL;  
}

uint8_t * enCODE264 :: framecodeSend(x264_t* pX264Handle, x264_picture_t* pPicIn,float * length)
{
	//* ������Ҫ�ĸ�������   
	int iNal = 0;  
	x264_nal_t *pNals = NULL;  
	x264_picture_t* pPicOut = new x264_picture_t;  
	x264_picture_init(pPicOut);  
	uint8_t * pSend = NULL;
	uint8_t * p = NULL;
	*length = 0;
	int step = 0;

	//����һ֡   
	int frame_size = x264_encoder_encode(pX264Handle,&pNals,&iNal,pPicIn,pPicOut);  

	if(frame_size >0)  
	{  
		for (int i = 0; i < iNal; ++i)  
		{
			*length = *length + pNals[i].i_payload;
		}  

		pSend = (uint8_t *)malloc(*length);
		p = pSend;

		//���������ݿ������ڴ���
		for (int i = 0; i < iNal; ++i)  
		{
			memcpy(p,pNals[i].p_payload,pNals[i].i_payload);
			step = pNals[i].i_payload;
			p = p+step;
		}  

		delete pPicOut;  
		pPicOut = NULL;  

		return pSend;
	}  

	delete pPicOut;  
	pPicOut = NULL;  

	return pSend;
}

x264_t* enCODE264 :: handleGet(IplImage * pFrame)
{
	x264_t* pX264Handle   = NULL;  

	/*videoShow avi;*/
	x264_param_t* pX264Param = new x264_param_t;  
	assert(pX264Param);  
	
	//* ���ò���   
	EncoderInit(pX264Param, pFrame->width, pFrame->height);

	//* �򿪱��������,ͨ��x264_encoder_parameters�õ����ø�X264�Ĳ���.ͨ��x264_encoder_reconfig����X264�Ĳ���   
	pX264Handle = x264_encoder_open(pX264Param);  
	assert(pX264Handle);

	delete pX264Param;  
	pX264Param = NULL;  

	return pX264Handle;
}

x264_picture_t * enCODE264 :: picInInit(IplImage * pFrame)
{
	x264_picture_t* pPicIn = NULL;
	
	//* ������Ҫ�ĸ�������   
	pPicIn = new x264_picture_t;  
	x264_picture_alloc(pPicIn, X264_CSP_I420, pFrame->width, pFrame->height); 
	pPicIn->img.i_csp = X264_CSP_I420;  /* yuv 4:2:0 planar */
	pPicIn->img.i_plane = 3;            /* Number of image planes: 3 ��ͼ��ƽ�� Y,U,V */
	
	return pPicIn;
}
	
void enCODE264 :: encodeOneFrame(IplImage * pFrame)
{
	unsigned char *RGB;
	unsigned char *YUV;
	RGB=(unsigned char *)malloc(pFrame->height*pFrame->width*3);
	YUV=(unsigned char *)malloc(pFrame->height*pFrame->width*1.5);

	//OPENCVͼƬ��ɫͨ����BGRתΪRGB
	cvBRGtoRGB(RGB, pFrame);
	
	//RGBתYUV
	Convert(RGB, YUV,pFrame->width,pFrame->height);//RGB to YUV

	//YUV3��ͨ��������ͼƬ��
	imgPalneCopy(pPicIn, YUV, pFrame);

	pPicIn->i_pts = frameNum;
	frameNum++;
	framecodeWrite(pX264Handle, pPicIn, pFile);
	
	free(RGB);
	free(YUV);
	RGB = NULL;
	YUV = NULL;
	
}

uint8_t * enCODE264 :: encodeOneFrame2(IplImage * pFrame, float * length, bool type)
{
	uint8_t * pSend = NULL;
	unsigned char *RGB;
	unsigned char *YUV;
	RGB=(unsigned char *)malloc(pFrame->height*pFrame->width*3);
	YUV=(unsigned char *)malloc(pFrame->height*pFrame->width*1.5);

	//OPENCVͼƬ��ɫͨ����BGRתΪRGB
	cvBRGtoRGB(RGB, pFrame);

	//RGBתYUV
	Convert(RGB, YUV,pFrame->width,pFrame->height);//RGB to YUV

	//YUV3��ͨ��������ͼƬ��
	imgPalneCopy(pPicIn, YUV, pFrame);

	pPicIn->i_pts = frameNum;
	frameNum++;
	
	if(true == type)
	{
		pSend = framecodeSend(pX264Handle, pPicIn,length);
	}
	if(false == type)
	{
		framecodeWrite(pX264Handle, pPicIn, pFile);
	}	

	free(RGB);
	free(YUV);
	RGB = NULL;
	YUV = NULL;

	return pSend;
}

int main(int argc, char** argv)  
{ 
	videoShow avi;
	bool ifRe = false;
	uint8_t * pSend = NULL;

	CvVideoWriter* writer = cvCreateVideoWriter(  
		"bak.avi",                               
		CV_FOURCC('D','X','5','0'),    
		5.0,
		avi.sizeGet()
		);
	avi.writer = writer;

	//* �����ļ�,���ڴ洢��������  
	char * h264Save = "saveOpencv.h264";
	FILE * pFile = fopen(h264Save, "wb"); 
	assert(pFile);  
	
	IplImage* pFrame = NULL; 
	pFrame = avi.currentFrameGet();
	if (NULL == pFrame)
	{
		printf("Failed to open camera!\n");  
		return -1;  
	}

	enCODE264 encodeFrame(pFrame,pFile);
	float length =0 ;
	while(1)
	{
		//encodeFrame.encodeOneFrame(pFrame);
		pSend = encodeFrame.encodeOneFrame2(pFrame, &length);
		//����������д���ļ�.
		fwrite(pSend, 1, length, pFile); 

		if (NULL != pFrame)
		{
			cvReleaseImage( &pFrame ); //�ͷ�ͼ��
		}
		
		pFrame = avi.currentFrameGet();
		int ifok = avi.aviSave(pFrame);

		if (NULL == pFrame)
		{
			printf("Failed to open camera!\n");  
			return -1;  
		}

		if ( (encodeFrame.getFrameNum() > 50) && (false ==ifRe) )
		{
			encodeFrame.reSet264(pFrame);
			ifRe = true;
			pFile = encodeFrame.getPFile();
		}

		if ( encodeFrame.getFrameNum() > 100)
		{
			break;
		}
	}

     return 0;  
}  

