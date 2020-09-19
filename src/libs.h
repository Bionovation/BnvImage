#ifndef _LIBS_H_83827384_
#define _LIBS_H_83827384_

// opencv 
#include "opencv2\opencv.hpp"
#ifdef _DEBUG
#pragma comment(lib, "opencv_world420d.lib")
#else
#pragma comment(lib, "opencv_world420.lib")
#endif // _DEBUG

// cuda
#include "cuda_runtime.h"


#endif // !_LIBS_H_83827384_
