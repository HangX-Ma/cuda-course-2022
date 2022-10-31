
#include "kernel.h"
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>



thrust::host_vector<int> adjTriNums_h_;
thrust::device_vector<int> adjTriNums_d_;
thrust::host_vector<int> prefix_sum_h_;
thrust::device_vector<int> prefix_sum_d_;

int* adjInfo;
REAL* gIntensityIn_d_;
REAL* gIntensityOut_d_;

int* gAdjTriNums_d_ptr;
int* gPrefix_sum_d_ptr;


extern std::vector<std::vector<int>> gAdjInfo;
extern std::vector<REAL> gIntensity[2];



__global__ void 
propagate_Kernel(int num_objects, REAL* curr, REAL* prev, int* adjObjNums, int* prefix_sum, int* adjInfo) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > num_objects - 1) {
        return;
    }
	int adjObjNum = adjObjNums[idx];

    curr[idx] = prev[idx];
    for (int i = 0; i < adjObjNum; i++) {
        curr[idx] += prev[adjInfo[prefix_sum[idx]/*offset*/ + i]];
    }
    curr[idx] /= (REAL)(adjObjNum + 1);
}

void doPropagateKernel(int flag, int num) {

    /* kernel property */
    int threadsPerBlock = 256;
    int blocksPerGrid = (num + threadsPerBlock - 1) / threadsPerBlock;
    
    if (flag) {
        cudaMemcpy(gIntensityIn_d_, gIntensity[flag].data(), sizeof(REAL) * num, cudaMemcpyHostToDevice);
        propagate_Kernel<<<blocksPerGrid, threadsPerBlock>>>
                (num, gIntensityOut_d_, gIntensityIn_d_, gAdjTriNums_d_ptr, gPrefix_sum_d_ptr, adjInfo);
    }
    else {
        cudaMemcpy(gIntensityOut_d_, gIntensity[flag].data(), sizeof(REAL) * num, cudaMemcpyHostToDevice);
        propagate_Kernel<<<blocksPerGrid, threadsPerBlock>>>
                (num, gIntensityIn_d_, gIntensityOut_d_, gAdjTriNums_d_ptr, gPrefix_sum_d_ptr, adjInfo);
    }
    cudaDeviceSynchronize();

    if (flag)
        cudaMemcpy(gIntensity[flag].data(), gIntensityOut_d_,  num * sizeof(REAL), cudaMemcpyDeviceToHost);
    else
        cudaMemcpy(gIntensity[flag].data(), gIntensityIn_d_,  num * sizeof(REAL), cudaMemcpyDeviceToHost);

}



void doGPUInit(int num, int flag) {
    int sum = 0;
    adjTriNums_h_.resize(num);
    prefix_sum_h_.resize(num);
    for (int i = 0; i < num; i++) {
        std::vector<int> &adjs = gAdjInfo[i];
        int size = adjs.size();
        adjTriNums_h_[i] = size;
        sum += size;
    }

    cudaMalloc((void**)&adjInfo, sum * sizeof(int));
    cudaMalloc((void**)&gIntensityIn_d_, num * sizeof(REAL));
    cudaMalloc((void**)&gIntensityOut_d_, num * sizeof(REAL));

    thrust::exclusive_scan(thrust::host, adjTriNums_h_.begin(), adjTriNums_h_.end(), prefix_sum_h_.begin(), 0);


    for (int i = 0; i < num; i++) {
        std::vector<int> &adjs = gAdjInfo[i];
        cudaMemcpy(adjInfo + prefix_sum_h_[i],
                    adjs.data(), 
                    sizeof(int) * adjTriNums_h_[i],
                    cudaMemcpyHostToDevice);
    }
    adjTriNums_d_ = adjTriNums_h_;
    prefix_sum_d_ = prefix_sum_h_;

    gAdjTriNums_d_ptr = adjTriNums_d_.data().get();
    gPrefix_sum_d_ptr = prefix_sum_d_.data().get();
}