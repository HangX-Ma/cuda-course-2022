#include "bvh.h"
#include "aabb.h"
#include "common.h"
#include <thrust/device_ptr.h>
#include <thrust/device_free.h>

#define HEAT_SOURCE_SIZE (3)


volatile int dstOut = -1;
float isource = 1.0f;
__device__ float dev_isource;

int heatSource[HEAT_SOURCE_SIZE] = {10, 100, 200};
int* dev_heatSource;


/* heat and color */
float* gIntensity_h_;
float* gIntensityIn_d_;
float* gIntensityOut_d_;

/* global value */
extern std::uint32_t gNumObjects;

lbvh::BVH* bvhInstance = lbvh::BVH::getInstance();
/* print info */
std::string div_signs(10, '-');

__global__ void 
propagate_Kernel(std::uint32_t num_objects, int* heatSource, std::uint32_t* adjObjects, 
        std::uint32_t* prefix_sum, std::uint32_t* adjObjNums, float *prev, float* curr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > num_objects - 1) {
        return;
    }
    std::uint32_t adjObjNum = adjObjNums[idx];

    curr[idx] = prev[idx];
    for (int i = 0; i < adjObjNum; i++) {
        curr[idx] += prev[adjObjects[prefix_sum[idx]/*offset*/ + i]];
    }
    curr[idx] /= (float)(adjObjNum + 1);

    /* keep source stable */
    for (int j = 0; j < HEAT_SOURCE_SIZE; j++) {
        if (idx == heatSource[j]) {
            curr[idx] = dev_isource;
        }
    }

}


__host__ void
lbvh::BVH::propagate() {
    if (bvh_status != lbvh::BVH_STATUS::STATE_PROPAGATE) {
        printf("PROPAGATE Kernel needs to be called at SATAE_PROPAGATE.\n");
        return;
    }

    /* get prefix sum pointer */
    std::uint32_t* scan_res_ptr = thrust::raw_pointer_cast(scan_res_d_.data());
    std::uint32_t* adjObjNumList_raw_ptr = thrust::raw_pointer_cast(adjObjNumList_d_.data());

    /* kernel property */
    int threadsPerBlock = 256;
    int blocksPerGrid = (gNumObjects + threadsPerBlock - 1) / threadsPerBlock;
    if (dstOut == 1) {
        propagate_Kernel<<<blocksPerGrid, threadsPerBlock>>>
                (gNumObjects, dev_heatSource, adjObjInfo_d_, scan_res_ptr, adjObjNumList_raw_ptr, gIntensityIn_d_, gIntensityOut_d_);
    }
    else if (dstOut == 0) {
        propagate_Kernel<<<blocksPerGrid, threadsPerBlock>>>
                (gNumObjects, dev_heatSource, adjObjInfo_d_, scan_res_ptr, adjObjNumList_raw_ptr, gIntensityOut_d_, gIntensityIn_d_);
    }
    HANDLE_ERROR(cudaDeviceSynchronize());

    /* copy out calculated value */
    if (dstOut == 1) {
        HANDLE_ERROR(cudaMemcpy(gIntensity_h_, gIntensityOut_d_,  gNumObjects * sizeof(float), cudaMemcpyDeviceToHost));
    } 
    else if (dstOut == 0) {
        HANDLE_ERROR(cudaMemcpy(gIntensity_h_, gIntensityIn_d_,  gNumObjects * sizeof(float), cudaMemcpyDeviceToHost));
    }

    /* swap in/out buffer */
    dstOut = 1 - dstOut;
}


void startHeatTransfer() {
    if (bvhInstance->getStatus() != lbvh::BVH_STATUS::STATE_PROPAGATE) {
        printf("Please complete prerequisites before propagating.\n");
        return;
    }
    std::cout << div_signs << "Start Heat Transfer" << div_signs << std::endl;

    if (dstOut == -1) {
        dstOut = 1;

        /* allocate data */
        gIntensity_h_ = (float*)malloc(gNumObjects * sizeof(float));
        HANDLE_ERROR(cudaMalloc((void**)&dev_heatSource, HEAT_SOURCE_SIZE * sizeof(int)));
        HANDLE_ERROR(cudaMalloc((void**)&gIntensityIn_d_, gNumObjects * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void**)&gIntensityOut_d_, gNumObjects * sizeof(float)));
        printf("--> Intensity memory has been allocated.\n");

        /* we need to copy the host value to deivce memory */
        HANDLE_ERROR(cudaMemcpyToSymbol(dev_isource, &isource, sizeof(float)));

        /* initialize the first iteration temperature */
        thrust::device_ptr<float> gIntensityIn_d_ptr(gIntensityIn_d_);
        thrust::for_each(thrust::device, 
            thrust::make_counting_iterator<std::uint32_t>(0),
            thrust::make_counting_iterator<std::uint32_t>(gNumObjects),
            [gIntensityIn_d_ptr] __device__ (std::uint32_t idx){
                gIntensityIn_d_ptr[idx] = 0.0f;
                return;
            });

        printf("--> Heat source IDs have been determined.\n");
        printf("--> Initialization done.\n");
        printf("--> propagating start...\n");
    }
    else {
        TIMING_BEGIN
        bvhInstance->propagate();
        TIMING_END("propagating ...")
    }
}


void quit_heatTransfer() {
    free(gIntensity_h_);
    HANDLE_ERROR(cudaMemcpyFromSymbol(&isource, dev_isource, sizeof(float)));
    HANDLE_ERROR(cudaFree(gIntensityIn_d_));
    HANDLE_ERROR(cudaFree(gIntensityOut_d_));
}
