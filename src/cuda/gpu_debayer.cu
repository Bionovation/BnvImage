
#include "cuda_runtime.h"

#include "gpu_util.h"


//默认宽高是偶数
// R G
// G B
__global__ void kernel_bayer_to_rgb(unsigned char *image_in, int width, int height, unsigned char* image_out)
{
    const int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const int thread_id = block_idx * blockDim.x + threadIdx.x;

    int image_size = width * height;
    if (thread_id >= image_size) return;

    int x = thread_id % width;
    int y = thread_id / width;

    int pos_in = (y * width + x);
    int pos_out = (y * width + x) * 3;

    int b = 0;
    int g = 0;
    int r = 0;


    if (y % 2 == 0) { //偶数行
        if (x % 2 == 0) { //偶数列
                          //R			
            r = image_in[pos_in];

            if (y == 0) {
                if (x == 0) {   //如果是0行0列
                    int b4 = image_in[(y + 1)*width + (x + 1)];
                    b = b4;

                    int g3 = image_in[(y + 1)*width + (x)];
                    int g4 = image_in[(y)*width + (x + 1)];
                    g = (g3 + g4) / 2;
                }
                else {
                    int b3 = image_in[(y + 1)*width + (x - 1)];
                    int b4 = image_in[(y + 1)*width + (x + 1)];
                    b = (b3 + b4) / 2;

                    int g2 = image_in[(y)*width + (x - 1)];
                    int g3 = image_in[(y + 1)*width + (x)];
                    int g4 = image_in[(y)*width + (x + 1)];
                    g = (g2 + g3 + g4) / 3;
                }
            }
            else {
                if (x == 0) {
                    int b1 = image_in[(y - 1)*width + (x + 1)];
                    int b4 = image_in[(y + 1)*width + (x + 1)];
                    b = (b1 + b4) / 2;

                    int g1 = image_in[(y - 1)*width + (x)];
                    int g2 = image_in[(y)*width + (x + 1)];
                    int g3 = image_in[(y + 1)*width + (x)];
                    g = (g1 + g3 + g2) / 3;

                }
                else {
                    int b1 = image_in[(y - 1)*width + (x - 1)];
                    int b2 = image_in[(y - 1)*width + (x + 1)];
                    int b3 = image_in[(y + 1)*width + (x - 1)];
                    int b4 = image_in[(y + 1)*width + (x + 1)];
                    b = (b1 + b2 + b3 + b4) / 4;

                    int g1 = image_in[(y - 1)*width + (x)];
                    int g2 = image_in[(y)*width + (x - 1)];
                    int g3 = image_in[(y + 1)*width + (x)];
                    int g4 = image_in[(y)*width + (x + 1)];
                    g = (g1 + g2 + g3 + g4) / 4;
                }
            }

        }
        else { //奇数列
               //G
            g = image_in[pos_in];

            if (y == 0) {
                if (x == width - 1) {
                    int b2 = image_in[(y + 1)*width + (x)];
                    b = (b2) / 1;

                    int r2 = image_in[(y)*width + (x - 1)];
                    r = (r2) / 1;
                }
                else {
                    int b2 = image_in[(y + 1)*width + (x)];
                    b = (b2) / 1;

                    int r1 = image_in[(y)*width + (x + 1)];
                    int r2 = image_in[(y)*width + (x - 1)];
                    r = (r1 + r2) / 2;
                }
            }
            else {
                if (x == width - 1) {
                    int b1 = image_in[(y - 1)*width + (x)];
                    int b2 = image_in[(y + 1)*width + (x)];
                    b = (b1 + b2) / 2;

                    int r2 = image_in[(y)*width + (x - 1)];
                    r = (r2) / 1;
                }
                else {
                    int b1 = image_in[(y - 1)*width + (x)];
                    int b2 = image_in[(y + 1)*width + (x)];
                    b = (b1 + b2) / 2;

                    int r1 = image_in[(y)*width + (x + 1)];
                    int r2 = image_in[(y)*width + (x - 1)];
                    r = (r1 + r2) / 2;
                }
            }
        }
    }
    else { //奇数行
        if (x % 2 == 0) { //偶数列
                          //G
            g = image_in[pos_in];
            if (y == height - 1) {
                if (x == 0) {
                    int b2 = image_in[(y)*width + (x + 1)];
                    b = (b2) / 1;

                    int r2 = image_in[(y - 1)*width + (x)];
                    r = (r2) / 1;
                }
                else {
                    int b1 = image_in[(y)*width + (x - 1)];
                    int b2 = image_in[(y)*width + (x + 1)];
                    b = (b1 + b2) / 2;


                    int r2 = image_in[(y - 1)*width + (x)];
                    r = (r2) / 1;

                }
            }
            else {
                if (x == 0) {
                    int b2 = image_in[(y)*width + (x + 1)];
                    b = (b2) / 1;

                    int r1 = image_in[(y + 1)*width + (x)];
                    int r2 = image_in[(y - 1)*width + (x)];
                    r = (r1 + r2) / 2;
                }
                else {
                    int b1 = image_in[(y)*width + (x - 1)];
                    int b2 = image_in[(y)*width + (x + 1)];
                    b = (b1 + b2) / 2;

                    int r1 = image_in[(y + 1)*width + (x)];
                    int r2 = image_in[(y - 1)*width + (x)];
                    r = (r1 + r2) / 2;
                }
            }


        }
        else { //奇数列
               //B
            b = image_in[pos_in];

            if (y == height - 1) {
                if (x == width - 1) {
                    int r1 = image_in[(y - 1)*width + (x - 1)];
                    r = (r1) / 1;

                    int g1 = image_in[(y - 1)*width + (x)];
                    int g2 = image_in[(y)*width + (x - 1)];
                    g = (g1 + g2) / 2;
                }
                else {
                    int r1 = image_in[(y - 1)*width + (x - 1)];
                    int r2 = image_in[(y - 1)*width + (x + 1)];
                    r = (r1 + r2) / 2;

                    int g1 = image_in[(y - 1)*width + (x)];
                    int g2 = image_in[(y)*width + (x - 1)];
                    int g4 = image_in[(y)*width + (x + 1)];
                    g = (g1 + g2 + g4) / 3;
                }
            }
            else {
                if (x == width - 1) {
                    int r1 = image_in[(y - 1)*width + (x - 1)];
                    int r3 = image_in[(y + 1)*width + (x - 1)];
                    r = (r1 + r3) / 2;

                    int g1 = image_in[(y - 1)*width + (x)];
                    int g2 = image_in[(y)*width + (x - 1)];
                    int g3 = image_in[(y + 1)*width + (x)];
                    g = (g1 + g2 + g3) / 3;
                }
                else {
                    int r1 = image_in[(y - 1)*width + (x - 1)];
                    int r2 = image_in[(y - 1)*width + (x + 1)];
                    int r3 = image_in[(y + 1)*width + (x - 1)];
                    int r4 = image_in[(y + 1)*width + (x + 1)];
                    r = (r1 + r2 + r3 + r4) / 4;

                    int g1 = image_in[(y - 1)*width + (x)];
                    int g2 = image_in[(y)*width + (x - 1)];
                    int g3 = image_in[(y + 1)*width + (x)];
                    int g4 = image_in[(y)*width + (x + 1)];
                    g = (g1 + g2 + g3 + g4) / 4;
                }
            }
        }
    }


    image_out[pos_out] = (unsigned char)r;
    image_out[pos_out + 1] = (unsigned char)g;
    image_out[pos_out + 2] = (unsigned char)b;
}

// R G
// G B
// 《Bayer图像色彩还原线性插值方法》文中的方法 3邻域
__global__ void kernel_debayer_to_rgb_n3(unsigned char *image_in, int width, int height, unsigned char* image_out)
{
    /*
    每个线程处理相邻的四个像素，rggb

    R:
    2j-1 2j   2j+1 2j+2
    2i-1
    2i   R    -    R
    2i+1 -    -
    2i+2 R         R

    G:
    2j-1 2j   2j+1 2j+2
    2i-1      G
    2i   G    -    G
    2i+1      G    -    G
    2i+2           G

    B:
    2j-1 2j   2j+1 2j+2
    2i-1 B         B
    2i        -    -
    2i+1 B    -    B
    2i+2

    */
    const int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const int thread_id = block_idx * blockDim.x + threadIdx.x;

    int x = (thread_id % (width / 2)) * 2;
    int y = (thread_id / (width / 2)) * 2;

    if (x >= width || y >= height) return;

    int pos_in = (y * width + x);
    int pos_out = (y * width + x) * 3;
    int spout = width * 3; // strap out

                           // 要取出的四个值
    unsigned char r[4] = { 0,0,0,0 };
    unsigned char g[6] = { 0,0,0,0,0,0 };
    unsigned char b[4] = { 0,0,0,0 };


    unsigned char& r00 = image_out[pos_out];
    unsigned char& r01 = image_out[pos_out + 3];
    unsigned char& r10 = image_out[pos_out + spout];
    unsigned char& r11 = image_out[pos_out + spout + 3];

    unsigned char& g00 = image_out[pos_out + 1];
    unsigned char& g01 = image_out[pos_out + 4];
    unsigned char& g10 = image_out[pos_out + spout + 1];
    unsigned char& g11 = image_out[pos_out + spout + 4];

    unsigned char& b00 = image_out[pos_out + 2];
    unsigned char& b01 = image_out[pos_out + 5];
    unsigned char& b10 = image_out[pos_out + spout + 2];
    unsigned char& b11 = image_out[pos_out + spout + 5];


    r[0] = image_in[pos_in];
    g[2] = image_in[pos_in + 1];
    g[3] = image_in[pos_in + width];
    b[3] = image_in[pos_in + width + 1];

    /*
    r[1] = image_in[pos_in + 2];
    r[2] = image_in[pos_in + 2*width];
    r[3] = image_in[pos_in + 2 * width + 2];

    g[0] = image_in[pos_in - width];
    g[1] = image_in[pos_in - 1];
    g[4] = image_in[pos_in + width + 2];
    g[5] = image_in[pos_in + 2*width + 1];

    b[0] = image_in[pos_in - width - 1];
    b[1] = image_in[pos_in - width + 1];
    b[2] = image_in[pos_in + width + 1];
    */

    // 中间
    if(y != 0 && x != 0 && y != height -2 && x != width - 2) {
        r[1] = image_in[pos_in + 2];
        r[2] = image_in[pos_in + 2 * width];
        r[3] = image_in[pos_in + 2 * width + 2];

        g[0] = image_in[pos_in - width];
        g[1] = image_in[pos_in - 1];
        g[4] = image_in[pos_in + width + 2];
        g[5] = image_in[pos_in + 2 * width + 1];

        b[0] = image_in[pos_in - width - 1];
        b[1] = image_in[pos_in - width + 1];
        b[2] = image_in[pos_in + width + 1];

        r00 = r[0];
        r01 = (r[0] + r[1]) / 2;
        r10 = (r[0] + r[2]) / 2;
        r11 = (r[0] + r[1] + r[2] + r[3]) / 4;

        g00 = (g[0] + g[1] + g[2] + g[3]) / 4;
        g01 = g[2];
        g10 = g[3];
        g11 = (g[4] + g[5] + g[2] + g[3]) / 4;

        b00 = (b[0] + b[1] + b[2] + b[3]) / 4;
        b01 = (b[1] + b[3]) / 2;
        b10 = (b[2] + b[3]) / 2;
        b11 = b[3];
    }
    // 边缘统一最近邻
    else {
        r00 = r01 = r10 = r11 = r[0];
        b00 = b01 = b10 = b11 = b[3];
        g00 = g11 = (g[2] + g[3]) / 2;
        g01 = g[2];
        g10 = g[3];
    }

    /*// 左上角
    else if (y == 0 && x == 0) {
        r[1] = image_in[pos_in + 2];
        r[2] = image_in[pos_in + 2 * width];
        r[3] = image_in[pos_in + 2 * width + 2];

        
        //g[0] = image_in[pos_in - width];
        //g[1] = image_in[pos_in - 1];
        g[4] = image_in[pos_in + width + 2];
        g[5] = image_in[pos_in + 2 * width + 1];

        //b[0] = image_in[pos_in - width - 1];
        //b[1] = image_in[pos_in - width + 1];
        //b[2] = image_in[pos_in + width + 1];
        
        r00 = r[0];
        r01 = (r[0] + r[1]) / 2;
        r10 = (r[0] + r[2]) / 2;
        r11 = (r[0] + r[1] + r[2] + r[3]) / 4;

        g00 = (g[2] + g[3]) / 2;
        g01 = g[2];
        g10 = g[3];
        g11 = (g[4] + g[5] + g[2] + g[3]) / 4;

        b00 = b01 = b10 = b11 = b[3];
    }
    // 右上角
    else if (y == 0 && x == width - 2) {
        //r[1] = image_in[pos_in + 2];
        r[2] = image_in[pos_in + 2 * width];
        //r[3] = image_in[pos_in + 2 * width + 2];

        //g[0] = image_in[pos_in - width];
        g[1] = image_in[pos_in - 1];
        //g[4] = image_in[pos_in + width + 2];
        g[5] = image_in[pos_in + 2 * width + 1];

        //b[0] = image_in[pos_in - width - 1];
        //b[1] = image_in[pos_in - width + 1];
        b[2] = image_in[pos_in + width + 1];

        //
        r00 = r[0];
        r01 = r[0];
        r10 = (r[0] + r[2]) / 2;
        r11 = (r[0] + r[2]) / 2;

        g00 = (g[1] + g[2] + g[3]) / 3;
        g01 = g[2];
        g10 = g[3];
        g11 = (g[5] + g[2] + g[3]) / 3;

        b00 = (b[2] + b[3]) / 2;
        b01 = b[3];
        b10 = (b[2] + b[3]) / 2;
        b11 = b[3];
    }
    // 左下角
    else if (y == height - 2 && x == 0) {
        r[1] = image_in[pos_in + 2];
        //r[2] = image_in[pos_in + 2 * width];
        //r[3] = image_in[pos_in + 2 * width + 2];

        g[0] = image_in[pos_in - width];
        //g[1] = image_in[pos_in - 1];
        g[4] = image_in[pos_in + width + 2];
        //g[5] = image_in[pos_in + 2 * width + 1];

        //b[0] = image_in[pos_in - width - 1];
        b[1] = image_in[pos_in - width + 1];
        //b[2] = image_in[pos_in + width + 1];

        //
        r00 = r[0];
        r01 = (r[0] + r[1]) / 2;
        r10 = r[0];
        r11 = (r[0] + r[1]) / 2;

        g00 = (g[0] + g[2] + g[3]) / 3;
        g01 = g[2];
        g10 = g[3];
        g11 = (g[4] + g[2] + g[3]) / 3;

        b00 = (b[1] + b[3]) / 2;
        b01 = (b[1] + b[3]) / 2;
        b10 = b[3];
        b11 = b[3];
    }

    // 右下角
    else if (y == height - 2 && x == width - 2) {
        //r[1] = image_in[pos_in + 2];
        //r[2] = image_in[pos_in + 2 * width];
        //r[3] = image_in[pos_in + 2 * width + 2];

        g[0] = image_in[pos_in - width];
        g[1] = image_in[pos_in - 1];
        //g[4] = image_in[pos_in + width + 2];
        //g[5] = image_in[pos_in + 2 * width + 1];

        b[0] = image_in[pos_in - width - 1];
        b[1] = image_in[pos_in - width + 1];
        b[2] = image_in[pos_in + width + 1];

        //
        r00 = r01 = r10 = r11 = r[0];

        g00 = (g[0] + g[1] + g[2] + g[3]) / 4;
        g01 = g[2];
        g10 = g[3];
        g11 = (g[2] + g[3]) / 2;

        b00 = (b[0] + b[1] + b[2] + b[3]) / 4;
        b01 = (b[1] + b[3]) / 2;
        b10 = (b[2] + b[3]) / 2;
        b11 = b[3];
    }*/
    
}

int gpu_bayer_to_rgb_n3(unsigned char *image_in, int width, int height, unsigned char *image_out, cudaStream_t stream)
{
    int image_size = width*height / 4;
    dim3 Db(THREADCOUNT); //Db.x==THREADCOUNT, Db.y==1, Db.z==1
    int blockCount = gpujpeg_div_and_round_up(image_size, Db.x);
    dim3 Dg = compute_grid_size(blockCount);

    if (stream != nullptr) {
        kernel_debayer_to_rgb_n3 << <Dg, Db, 0, stream >> > (
            image_in,
            width,
            height,
            image_out
            );
    }
    else {
        kernel_debayer_to_rgb_n3 << <Dg, Db >> > (
            image_in,
            width,
            height,
            image_out
            );
    }


    gpujpeg_cuda_check_error("GPU RECOLOR FAILED.", return -1);


    return 0;
}


int gpu_bayer_to_rgb(unsigned char *image_in, int width, int height, unsigned char *image_out)
{
    int image_size = width*height;
    dim3 Db(THREADCOUNT); //Db.x==THREADCOUNT, Db.y==1, Db.z==1
    int blockCount = gpujpeg_div_and_round_up(image_size, Db.x);
    dim3 Dg = compute_grid_size(blockCount);

    kernel_bayer_to_rgb << <Dg, Db >> > (
        image_in,
        width,
        height,
        image_out
        );
    //cudaStreamSynchronize(0);
    gpujpeg_cuda_check_error("GPU RECOLOR FAILED.", return -1);

    return 0;
}