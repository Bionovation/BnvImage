#include "cuda_runtime.h"

int gpu_bayer_to_rgb_n3(unsigned char *image_in, int width, int height, unsigned char *image_out, cudaStream_t stream);

int gpu_bayer_to_rgb(unsigned char *image_in, int width, int height, unsigned char *image_out);