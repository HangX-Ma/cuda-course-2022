#include "bvh.h"
#include "aabb.h"
#include "common.h"

#define HEAT_SOURCE_SIZE (3)


volatile int dstOut = -1;
float initial_val = 1.0f;
int heatSource[HEAT_SOURCE_SIZE] = {10, 100, 200};

/* heat and color */
float* gIntensity_h_;
float* gIntensityIn_d_;
float* gIntensityOut_d_;

/* global value */
extern std::uint32_t gNumObjects;
extern std::uint32_t gNumAdjObjects;

lbvh::BVH* bvhInstance = lbvh::BVH::getInstance();
/* print info */
std::string div_signs(10, '-');


__global__ void 
propagate_Kernel(std::uint32_t num_objects, std::uint32_t** adjObjects, std::uint32_t* adjObjNums, float *prev, float* curr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > num_objects - 1) {
        return;
    }

    std::uint32_t* adjTriPtr = adjObjects[idx];
    std::uint32_t adjObjNum = adjObjNums[idx];

    curr[idx] = prev[idx];
    for (int i = 0; i < adjObjNum; i++) {
        curr[idx] += prev[adjTriPtr[i]];
    }
    curr[idx] /= static_cast<float>(adjObjNum + 1);
}


__host__ void
lbvh::BVH::propagate() {
    if (bvh_status != lbvh::BVH_STATUS::STATE_PROPAGATE) {
        printf("PROPAGATE Kernel needs to be called at SATAE_PROPAGATE.\n");
        return;
    }

    float *in_d_, *out_d_;
    if (dstOut == 1) {
        in_d_  = gIntensityIn_d_;
        out_d_ = gIntensityOut_d_;
    }
    else {
        out_d_ = gIntensityIn_d_;
        in_d_  = gIntensityOut_d_;
    }

    /* kernel property */
    int threadsPerBlock = 256;
    int blocksPerGrid = (gNumObjects - 1 + threadsPerBlock - 1) / threadsPerBlock;
    propagate_Kernel<<<blocksPerGrid, threadsPerBlock>>>(gNumObjects, adjObjInfo_d_, adjObjNum_d_, in_d_, out_d_);
    HANDLE_ERROR(cudaDeviceSynchronize());
    
    /* keep source constant */
    for (int i = 0; i < HEAT_SOURCE_SIZE; i++) {
        cudaMemcpy(gIntensityIn_d_ + heatSource[i], &initial_val, sizeof(float), cudaMemcpyHostToDevice);
    }
    
    /* copy out calculated value */
    if (dstOut == 1) {
        cudaMemcpy(gIntensity_h_, gIntensityOut_d_,  num_adjObjects * sizeof(float), cudaMemcpyDeviceToHost);
    } 
    else {
        cudaMemcpy(gIntensity_h_, gIntensityIn_d_,  num_adjObjects * sizeof(float), cudaMemcpyDeviceToHost);
    }
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
        gIntensity_h_ = (float*)malloc(gNumAdjObjects * sizeof(float));
        cudaMalloc((void**)&gIntensityIn_d_, gNumAdjObjects * sizeof(float));
        cudaMalloc((void**)&gIntensityOut_d_, gNumAdjObjects * sizeof(float));
        printf("--> Intensity memory has been allocated.\n");

        /* initialize the first iteration temperature */
        float initial_zero = 0.0f;
        for (int i = 0; i < gNumAdjObjects; i++) {
            cudaMemcpy(gIntensityIn_d_ + i, &initial_zero, sizeof(float), cudaMemcpyHostToDevice);
        }

        for (int i = 0; i < HEAT_SOURCE_SIZE; i++) {
            cudaMemcpy(gIntensityIn_d_ + heatSource[i], &initial_val, sizeof(float), cudaMemcpyHostToDevice);
        }
        printf("--> Heat source IDs have been determined. Initialization done.\n");
        printf("--> propagating start...\n");
    }
    else {
        TIMING_BEGIN
        bvhInstance->propagate();
        TIMING_END("propagating ...")
    }
}

