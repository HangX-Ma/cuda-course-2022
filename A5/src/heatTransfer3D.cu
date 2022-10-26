#include "bvh.h"
#include "query.h"
#include "aabb.h"
#include "book.h"

#include <cuda.h>
#include <thrust/host_vector.h>

#define HEAT_SOURCE_SIZE (3)


extern std::uint32_t gNumObjects;
lbvh::BVH* bvhInstancePtr = lbvh::BVH::getInstance();

volatile bool dstOut = NULL;
float initial_val = 1.0f;
int heatSource[HEAT_SOURCE_SIZE] = {10, 100, 200};

thrust::host_vector<float> gIntensity_h_;
thrust::device_vector<float> gIntensityIn_d_;
thrust::device_vector<float> gIntensityOut_d_;

__global__ void 
propagate_kernel(std::uint32_t** adjObjects, std::uint32_t* adjObjNums, float *prev, float* curr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > gNumObjects - 1) {
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
    if (bvh_status != BVH_STATUS::STATE_PROPAGATE) {
        printf("PROPAGATE Kernel needs to be called at SATAE_PROPAGATE.\n");
        return;
    }

    float *in_d_, *out_d_;
    if (dstOut) {
        in_d_  = gIntensityIn_d_.data().get();
        out_d_ = gIntensityOut_d_.data().get();
    }
    else {
        out_d_ = gIntensityIn_d_.data().get();
        in_d_  = gIntensityOut_d_.data().get();
    }

    /* kernel property */
    int threadsPerBlock = 256;
    int blocksPerGrid = (gNumObjects - 1 + threadsPerBlock - 1) / threadsPerBlock;
    propagate_kernel<<<blocksPerGrid, threadsPerBlock>>>(adjObjInfo_d_, adjObjNum_d_, in_d_, out_d_);
    HANDLE_ERROR(cudaDeviceSynchronize());
    /* keep source constant */
    for (int i = 0; i < HEAT_SOURCE_SIZE; i++) {
        cudaMemcpy(gIntensityIn_d_.data().get() + heatSource[i], &initial_val, sizeof(float), cudaMemcpyHostToDevice);
    }
    
    /* copy out calculated value */
    if (dstOut) {
        gIntensity_h_ = gIntensityOut_d_;
    } 
    else {
        gIntensity_h_ = gIntensityIn_d_;
    }
    dstOut = !dstOut;
}


void startHeatTransfer() {
    if (bvhInstancePtr->getStatus() != lbvh::BVH_STATUS::STATE_PROPAGATE) {
        printf("Please complete prerequisites before propagating.\n");
        return;
    }

    if (dstOut == NULL) {
        dstOut = true;

        std::uint32_t num_adjObjects = bvhInstancePtr->getAdjObjectNum();

        /* allocate data */
        gIntensity_h_.resize(num_adjObjects, 0);

        gIntensityIn_d_.resize(num_adjObjects, 0);
        gIntensityOut_d_.resize(num_adjObjects, 0);

        for (int i = 0; i < HEAT_SOURCE_SIZE; i++) {
            cudaMemcpy(gIntensityIn_d_.data().get() + heatSource[i], &initial_val, sizeof(float), cudaMemcpyHostToDevice);
        }
    }
    else {
        bvhInstancePtr->propagate();
    }
}

