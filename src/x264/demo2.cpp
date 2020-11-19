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



#define FAIL_IF_ERROR( cond, ... )\
do\
{\
    if( cond )\
    {\
        fprintf( stderr, __VA_ARGS__ );\
        goto fail;\
    }\
} while( 0 )

int main(int argc, char **argv)
{
	int width, height;
	x264_param_t param;
	x264_picture_t pic;
	x264_picture_t pic_out;
	x264_t *h;
	int i_frame = 0;
	int i_frame_size;
	x264_nal_t *nal;
	int i_nal;
	FILE *file_in = NULL;
	FILE *file_out = NULL;


	FAIL_IF_ERROR(!(argc > 3), "Example usage: example <352x288> <input.yuv> <output.h264>\n");
	FAIL_IF_ERROR(2 != sscanf(argv[1], "%dx%d", &width, &height), "resolution not specified or incorrect\n");

	const char *in_filename = argv[2];
	const char *out_filename = argv[3];
	file_in = fopen(in_filename, "r+b");
	if (!file_in) {
		fprintf(stderr, "open input failed, %s\n", in_filename);
		goto fail;
	}
	file_out = fopen(out_filename, "w+b");
	if (!file_out) {
		fprintf(stderr, "open output failed, %s\n", out_filename);
		goto fail;
	}
	/* Get default params for preset/tuning */
	if (x264_param_default_preset(&param, "medium", NULL) < 0)
		goto fail;

	/* Configure non-default params */
	// param.i_bitdepth = 8;
	param.i_csp = X264_CSP_I420;
	param.i_width = width;
	param.i_height = height;
	param.b_vfr_input = 0;
	param.b_repeat_headers = 1;
	param.b_annexb = 1;

	/* Apply profile restrictions. */
	if (x264_param_apply_profile(&param, "high") < 0)
		goto fail;

	if (x264_picture_alloc(&pic, param.i_csp, param.i_width, param.i_height) < 0)
		goto fail;
#undef fail
#define fail fail2

	h = x264_encoder_open(&param);
	if (!h)
		goto fail;
#undef fail
#define fail fail3

	int luma_size = width * height;
	int chroma_size = luma_size / 4;
	/* Encode frames */
	for (;; i_frame++)
	{
		/* Read input frame */
		if (fread(pic.img.plane[0], 1, luma_size, file_in) != luma_size)
			break;
		if (fread(pic.img.plane[1], 1, chroma_size, file_in) != chroma_size)
			break;
		if (fread(pic.img.plane[2], 1, chroma_size, file_in) != chroma_size)
			break;

		pic.i_pts = i_frame;
		i_frame_size = x264_encoder_encode(h, &nal, &i_nal, &pic, &pic_out);
		if (i_frame_size < 0)
			goto fail;
		else if (i_frame_size)
		{
			if (!fwrite(nal->p_payload, i_frame_size, 1, file_out))
				goto fail;
		}
	}
	/* Flush delayed frames */
	while (x264_encoder_delayed_frames(h))
	{
		i_frame_size = x264_encoder_encode(h, &nal, &i_nal, NULL, &pic_out);
		if (i_frame_size < 0)
			goto fail;
		else if (i_frame_size)
		{
			if (!fwrite(nal->p_payload, i_frame_size, 1, file_out))
				goto fail;
		}
	}

	x264_encoder_close(h);
	x264_picture_clean(&pic);
	if (file_in)
		fclose(file_in);
	if (file_out)
		fclose(file_out);
	return 0;

#undef fail
	fail3 :
		  x264_encoder_close(h);
	  fail2:
		  x264_picture_clean(&pic);
	  fail:
		  if (file_in)
			  fclose(file_in);
		  if (file_out)
			  fclose(file_out);
		  return -1;
}