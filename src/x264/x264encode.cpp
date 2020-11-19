
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



//转换矩阵
#define MY(a,b,c) (( a*  0.2989  + b*  0.5866  + c*  0.1145))
#define MU(a,b,c) (( a*(-0.1688) + b*(-0.3312) + c*  0.5000 + 128))
#define MV(a,b,c) (( a*  0.5000  + b*(-0.4184) + c*(-0.0816) + 128))

//大小判断
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
	
	//默认为读取摄像头视频
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

//如果是摄像头，获取这个时刻的帧；如果是视频，获取一帧，返回的pFrameOut使用完之后要释放，否则会内存泄漏
IplImage * videoShow :: currentFrameGet()
{
	if(NULL == pCapture)
	{
		fprintf(stderr, "Can not get currentFrame.\n");
		return NULL;
	}

	IplImage* pFrame = NULL; 
	IplImage* pFrameOut = NULL;

	//函数cvQueryFrame从摄像头或者文件中抓取一帧，然后解压并返回这一帧。这个函数仅仅是函数cvGrabFrame和函数cvRetrieveFrame在一起调用的组合。返回的图像不可以被用户释放或者修改
	pFrame = cvQueryFrame( pCapture );

	pFrameOut = cvCreateImage(cvSize(pFrame->width,pFrame->height),pFrame->depth, pFrame->nChannels);
	if (NULL == pFrameOut)
	{
		fprintf(stderr, "Can not create frame.\n");
		return NULL;
	}

	//cvCloneImage函数：这个函数已验证会出现较大的内存泄露！！虽然可以释放，但因程序复杂不知道在那里释放，因为它每次拷贝是制作图像的完整拷贝包括头、ROI和数据。每次使用时编译器会分配新的内存空间，不会覆盖以前的内容。一个752*480大小或是稍小的图像，每次泄露的内存大约为1M。
	//解决方法：使用cvCopy函数代替
	cvCopy(pFrame,pFrameOut,NULL);  
	pFrameNum++;

	return pFrameOut;
}

//获取当前帧号
int videoShow :: currentFrameNum()
{
	return pFrameNum;
}

//把当前帧保存到avi视频中
bool videoShow :: aviSave(IplImage* p)
{
	if ((NULL == writer) || (NULL == p))
	{
		return false;
	}
	cvWriteToAVI(writer, p);
	return true;
}

//保存图片，如果没有输入图片名，则默认保存为resultPic.jpg
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

//获取argc，argv参数
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

	//打开视频文件
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

//读取视频
void videoShow :: videoOpen(char* videoName)
{
	if(NULL != pCapture)
	{
		cvReleaseCapture(&pCapture);
		CvCapture* pCapture = NULL;
	}
	
	//打开视频文件
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

	//初始化句柄
	pX264Handle = handleGet(pFrame);

	//初始化
	pPicIn = picInInit(pFrame);

	if (NULL == pF)
	{
		//* 创建默认文件,用于存储编码数据   
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
	//* 清除图像区域   
	if (NULL != pPicIn)
	{
		x264_picture_clean(pPicIn);  
		delete pPicIn;  
		pPicIn = NULL; 
	}

	if (NULL != pX264Handle)
	{
		//* 关闭编码器句柄   
		x264_encoder_close(pX264Handle);  
		pX264Handle = NULL;  
	}

	frameNum = 0;

	//初始化句柄
	pX264Handle = handleGet(pFrame);

	//初始化
	pPicIn = picInInit(pFrame);

	if (NULL == pF)
	{
		//* 创建默认文件,用于存储编码数据   
		pFile = fopen("opencv.h264", "wb"); 
		assert(pFile); 
		return;
	}
	pFile = pF;

}

enCODE264 :: ~enCODE264()
{
	//* 清除图像区域   
	if (NULL != pPicIn)
	{
		x264_picture_clean(pPicIn);  
		delete pPicIn;  
		pPicIn = NULL; 
	}

	if (NULL != pX264Handle)
	{
		//* 关闭编码器句柄   
		x264_encoder_close(pX264Handle);  
		pX264Handle = NULL;  
	}
}

int enCODE264 :: getFrameNum()
{
	return frameNum;
}

//* 配置参数   
void enCODE264 :: EncoderInit(x264_param_t* pX264Param, int width, int height)
{
	//* 使用默认参数，在这里因为我的是实时网络传输，所以我使用了zerolatency的选项，使用这个选项之后就不会有delayed_frames，如果你使用的不是这样的话，还需要在编码完成之后得到缓存的编码帧   
	x264_param_default_preset(pX264Param, "veryfast", "zerolatency");  

	//* cpuFlags   
	pX264Param->i_threads  = X264_SYNC_LOOKAHEAD_AUTO;//* 取空缓冲区继续使用不死锁的保证.   
	//* 视频选项      
	pX264Param->i_width   = width; //* 要编码的图像宽度.   
	pX264Param->i_height  = height; //* 要编码的图像高度   
	pX264Param->i_frame_total = 0; //* 编码总帧数.不知道用0.   
	pX264Param->i_keyint_max = 10;   

	//* 流参数   
	pX264Param->i_bframe  = 5;  
	pX264Param->b_open_gop  = 0;  
	pX264Param->i_bframe_pyramid = 0;  
	pX264Param->i_bframe_adaptive = X264_B_ADAPT_TRELLIS; 

	//* Log参数，不需要打印编码信息时直接注释掉就行   
	pX264Param->i_log_level  = X264_LOG_DEBUG;  

	//* 速率控制参数   
	pX264Param->rc.i_bitrate = 1024 * 10;//* 码率(比特率,单位Kbps)  

	//* muxing parameters   
	pX264Param->i_fps_den  = 1; //* 帧率分母   
	pX264Param->i_fps_num  = 10;//* 帧率分子   
	pX264Param->i_timebase_den = pX264Param->i_fps_num;  
	pX264Param->i_timebase_num = pX264Param->i_fps_den;  

	//* 设置Profile.使用Baseline profile   
	x264_param_apply_profile(pX264Param, x264_profile_names[0]); 
	return;
}

/*-----------------按照色度空间分配YUV420类型内存，并返回内存的首地址作为指针---------------
我们常说得YUV420属于planar格式的YUV，使用三个数组分开存放YUV三个分量，就像是
一个三维平面一样。在常见H264测试的YUV序列中,例如CIF图像大小的YUV序列(352*288),在
文件开始并没有文件头,直接就是YUV数据,先存第一帧的Y信息,长度为352*288个byte, 然后是第
一帧U信息长度是352*288/4个byte, 最后是第一帧的V信息,长度是352*288/4个byte, 因此可以算
出第一帧数据总长度是352*288*1.5,即152064个byte, 如果这个序列是300帧的话, 那么序列总
长度即为152064*300=44550KB,这也就是为什么常见的300帧CIF序列总是44M的原因.
---------------------------------------------------------------------------------*/
void enCODE264 :: Convert(unsigned char *RGB, unsigned char *YUV,unsigned int width,unsigned int height)
{
	//变量声明
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
				//上面i仍有效
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
		//把YUV分量数据分别拷贝到三个平面，三个plane[i]数组分开存放YUV三个分量
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
	//pFrame中BGR转RGB
	for(int i=0;i<pFrame->height;i++)
	{
		for(int j=0;j<pFrame->width;j++)
		{
			//从opencv的pFrame中取位图RGB数据,并把BGR转RGB
			RGB[(i*pFrame->width+j)*3]   = pFrame->imageData[i * pFrame->widthStep + j * 3 + 2];
			RGB[(i*pFrame->width+j)*3+1] = pFrame->imageData[i * pFrame->widthStep + j * 3 + 1];
			RGB[(i*pFrame->width+j)*3+2] = pFrame->imageData[i * pFrame->widthStep + j * 3  ]; 
		}
	}
}

void enCODE264 :: framecodeWrite(x264_t* pX264Handle, x264_picture_t* pPicIn,FILE * pFile)
{
	//* 编码需要的辅助变量   
	int iNal = 0;  
	x264_nal_t *pNals = NULL;  
	x264_picture_t* pPicOut = new x264_picture_t;  
	x264_picture_init(pPicOut);  

	//编码一帧   
	int frame_size = x264_encoder_encode(pX264Handle,&pNals,&iNal,pPicIn,pPicOut);  

	if(frame_size >0)  
	{  
		for (int i = 0; i < iNal; ++i)  
		{
			//将编码数据写入文件.   
			fwrite(pNals[i].p_payload, 1, pNals[i].i_payload, pFile);  
		}  
	}  

	delete pPicOut;  
	pPicOut = NULL;  
}

uint8_t * enCODE264 :: framecodeSend(x264_t* pX264Handle, x264_picture_t* pPicIn,float * length)
{
	//* 编码需要的辅助变量   
	int iNal = 0;  
	x264_nal_t *pNals = NULL;  
	x264_picture_t* pPicOut = new x264_picture_t;  
	x264_picture_init(pPicOut);  
	uint8_t * pSend = NULL;
	uint8_t * p = NULL;
	*length = 0;
	int step = 0;

	//编码一帧   
	int frame_size = x264_encoder_encode(pX264Handle,&pNals,&iNal,pPicIn,pPicOut);  

	if(frame_size >0)  
	{  
		for (int i = 0; i < iNal; ++i)  
		{
			*length = *length + pNals[i].i_payload;
		}  

		pSend = (uint8_t *)malloc(*length);
		p = pSend;

		//将编码数据拷贝到内存中
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
	
	//* 配置参数   
	EncoderInit(pX264Param, pFrame->width, pFrame->height);

	//* 打开编码器句柄,通过x264_encoder_parameters得到设置给X264的参数.通过x264_encoder_reconfig更新X264的参数   
	pX264Handle = x264_encoder_open(pX264Param);  
	assert(pX264Handle);

	delete pX264Param;  
	pX264Param = NULL;  

	return pX264Handle;
}

x264_picture_t * enCODE264 :: picInInit(IplImage * pFrame)
{
	x264_picture_t* pPicIn = NULL;
	
	//* 编码需要的辅助变量   
	pPicIn = new x264_picture_t;  
	x264_picture_alloc(pPicIn, X264_CSP_I420, pFrame->width, pFrame->height); 
	pPicIn->img.i_csp = X264_CSP_I420;  /* yuv 4:2:0 planar */
	pPicIn->img.i_plane = 3;            /* Number of image planes: 3 个图像平面 Y,U,V */
	
	return pPicIn;
}
	
void enCODE264 :: encodeOneFrame(IplImage * pFrame)
{
	unsigned char *RGB;
	unsigned char *YUV;
	RGB=(unsigned char *)malloc(pFrame->height*pFrame->width*3);
	YUV=(unsigned char *)malloc(pFrame->height*pFrame->width*1.5);

	//OPENCV图片颜色通道由BGR转为RGB
	cvBRGtoRGB(RGB, pFrame);
	
	//RGB转YUV
	Convert(RGB, YUV,pFrame->width,pFrame->height);//RGB to YUV

	//YUV3个通道拷贝到图片中
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

	//OPENCV图片颜色通道由BGR转为RGB
	cvBRGtoRGB(RGB, pFrame);

	//RGB转YUV
	Convert(RGB, YUV,pFrame->width,pFrame->height);//RGB to YUV

	//YUV3个通道拷贝到图片中
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

	//* 创建文件,用于存储编码数据  
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
		//将编码数据写入文件.
		fwrite(pSend, 1, length, pFile); 

		if (NULL != pFrame)
		{
			cvReleaseImage( &pFrame ); //释放图像
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

