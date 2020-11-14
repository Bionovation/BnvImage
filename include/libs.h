#ifndef _LIBS_H_83827384_
#define _LIBS_H_83827384_

// opencv 
#include "opencv2\opencv.hpp"
#include "opencv2\imgproc\types_c.h"
#include "opencv2\imgproc\imgproc_c.h"
#ifdef _DEBUG
#pragma comment(lib, "opencv_world420d.lib")
//#pragma comment(lib, "opencv_world341d.lib")
#else
#pragma comment(lib, "opencv_world420.lib")
//#pragma comment(lib, "opencv_world341.lib")
#endif // _DEBUG

// cuda
#include "cuda_runtime.h"


#endif // !_LIBS_H_83827384_
