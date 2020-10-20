#include <iostream>
#include <time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

void cumsum_cpu(double *in, double *out, int length)
{
    clock_t start = clock();
    if (length < 1) return;
    out[0] = in[0];
    for (int i = 1; i < length; i++) {
        out[i] = in[i] + out[i - 1];
    }
    clock_t end = clock();

    cout << "cpu time spend: "
        << end - start
        << " ms"
        << endl;
    return;
}


// 简单求和
__global__ void kernel_simpleAdd(double *in, double *out, int length)
{
    if (length < 1) return;
    out[0] = in[0];
    for (int i = 1; i < length; i++) {
        out[i] = in[i] + out[i - 1];
    }
}

// 简单求和-并行
__global__ void kernel_simpleAdd2(double **inarr, double **outarr, int *lengtharr, int arrsize)
{
    int r = blockIdx.x;
    if (r >= arrsize) return;
    int length = lengtharr[r];
    double *in = inarr[r];
    double *out = outarr[r];

    if (length < 1) return;

    out[0] = in[0];
    for (int i = 1; i < length; i++) {
        out[i] = in[i] + out[i - 1];
    }
}

void cumsum_gpu(double *in, double *out, int length)
{
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    size_t sz = length * sizeof(double);
    double *in_dev, *out_dev;
    cudaMalloc(&in_dev, sz);
    cudaMalloc(&out_dev, sz);

    
    cudaMemcpy(in_dev, in, sz, cudaMemcpyHostToDevice);

    cudaEventRecord(ev0);
    kernel_simpleAdd << <1, 1 >> >(in_dev, out_dev, length);
    cudaEventRecord(ev1);

    cudaMemcpy(out, out_dev,sz, cudaMemcpyDeviceToHost);
    
    cudaStreamSynchronize(0);

    float tp = 0;
    cudaEventElapsedTime(&tp, ev0, ev1);

    cout << "gpu time spend: "
        << tp
        << " ms"
        << endl;
}


void cumsum_gpu2(double *in, double *out, int length)
{
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    int arrsz = 4;
    int sz = length * sizeof(double);
    int *length_arr = new int[arrsz];
    double **in_dev_arr = new double*[arrsz];
    double **out_dev_arr = new double*[arrsz];

    for (int i = 0; i < arrsz; i++) {
        length_arr[i] = length;
        cudaMalloc(&in_dev_arr[i], sz);
        cudaMalloc(&out_dev_arr[i], sz);
        cudaMemcpy(in_dev_arr[i], in, sz, cudaMemcpyHostToDevice);
    }

    int *length_arr_dev = nullptr;
    double **in_dev_arr_dev = nullptr;
    double **out_dev_arr_dev = nullptr;
    cudaMalloc(&length_arr_dev, sizeof(int) * arrsz);
    cudaMalloc(&in_dev_arr_dev, sizeof(void*) * arrsz);
    cudaMalloc(&out_dev_arr_dev, sizeof(void*) * arrsz);

    cudaMemcpy(length_arr_dev, length_arr, sizeof(int) * arrsz, cudaMemcpyHostToDevice);
    cudaMemcpy(in_dev_arr_dev, in_dev_arr, sizeof(void*) * arrsz, cudaMemcpyHostToDevice);
    cudaMemcpy(out_dev_arr_dev, out_dev_arr, sizeof(void*) * arrsz, cudaMemcpyHostToDevice);

    cudaEventRecord(ev0);
    kernel_simpleAdd2 << <1, arrsz >> >(in_dev_arr_dev, out_dev_arr_dev, length_arr_dev, arrsz);
    cudaEventRecord(ev1);


    cudaStreamSynchronize(0);

    float tp = 0;
    cudaEventElapsedTime(&tp, ev0, ev1);

    cout << "gpu time spend-2: "
        << tp
        << " ms"
        << endl;
}



int main() 
{
    int length = 20000;
    double *arr = new double[length];
    double *sum = new double[length];
    double *sum2 = new double[length];

    for (int i = 0; i < length; i++)
    {
        arr[i] = i;
    }

    cin.ignore();


    // cpu
    cumsum_cpu(arr, sum, length);

    // gpu
    cumsum_gpu(arr, sum2, length);

    // gpu 多个数组同时处理
    cumsum_gpu2(arr, sum2, length);

    

    for (int i = 0; i < length; i++)
    {
        double err = sum[i] - sum2[i];
        if (fabs(err) > 1E-8) {
            cout << err << endl;
        }
        
    }
    
    cin.ignore();

    return 0;
}