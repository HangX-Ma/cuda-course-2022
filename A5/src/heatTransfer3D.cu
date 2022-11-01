#include "bvh.h"
#include "aabb.h"
#include "common.h"
#include <thrust/device_ptr.h>
#include <thrust/device_free.h>

#define NORMAL_GPU (1)
#define QUICK_TRANS (1)

#define HEAT_SOURCE_SIZE (3)
#define HEAT_TRANSFER_SPEED (0.05f)

volatile int dstOut = -1;
float isource = 1.0f;

int heatSource[HEAT_SOURCE_SIZE] = {100, 10, 200};


/* heat and color */
float* gIntensity_h_; // pined memory

#if NORMAL_GPU
float* gIntensityIn_d_;
float* gIntensityOut_d_;
#else
#endif

/* global value */
extern std::uint32_t gNumObjects;
extern std::uint32_t* gSortedObjIDs;

lbvh::BVH* bvhInstance = lbvh::BVH::getInstance();
/* print info */
std::string div_signs(10, '-');

#if QUICK_TRANS
__global__ void 
propagate_Kernel(std::uint32_t num_objects, std::uint32_t* adjObjects, std::uint32_t* sortedIDs,
        std::uint32_t* prefix_sum, std::uint32_t* adjObjNums, float *prev, float* curr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > num_objects - 1) {
        return;
    }
    std::uint32_t adjObjNum = adjObjNums[idx];

    curr[idx] = 0.6 * prev[idx]; // heat loss
    for (int i = 0; i < adjObjNum; i++) {
        curr[idx] += prev[adjObjects[prefix_sum[idx]/*offset*/ + i]];
    }
    curr[idx] /= (float)(adjObjNum + 1);
    curr[idx] += HEAT_TRANSFER_SPEED * prev[idx];
    curr[idx] = fminf(curr[idx], 1.0f);
}
#else
__global__ void 
propagate_Kernel(std::uint32_t num_objects, std::uint32_t* adjObjects, std::uint32_t* sortedIDs,
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
}
#endif

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
    #if NORMAL_GPU
    HANDLE_ERROR(cudaMemcpy(gIntensityIn_d_, gIntensity_h_, sizeof(float) * gNumObjects, cudaMemcpyHostToDevice));
    propagate_Kernel<<<blocksPerGrid, threadsPerBlock>>>
            (gNumObjects, adjObjInfo_d_, gSortedObjIDs, scan_res_ptr, adjObjNumList_raw_ptr, gIntensityIn_d_, gIntensityOut_d_);
    HANDLE_ERROR(cudaDeviceSynchronize());
    /* copy out calculated value */
    HANDLE_ERROR(cudaMemcpy(gIntensity_h_, gIntensityOut_d_,  gNumObjects * sizeof(float), cudaMemcpyDeviceToHost));
    #else


    #endif


    for (int i = 0; i < HEAT_SOURCE_SIZE; i++) {
        gIntensity_h_[heatSource[i]] = 1.0;
    }

    /* swap in/out buffer */
    dstOut = 1 - dstOut;
}


void startHeatTransfer() {
    if (bvhInstance->getStatus() != lbvh::BVH_STATUS::STATE_PROPAGATE) {
        printf("Please complete prerequisites before propagating.\n");
        return;
    }
    if (dstOut == -1) {
        dstOut = 1;

        cudaDeviceProp  prop;
        int whichDevice;
        HANDLE_ERROR( cudaGetDevice( &whichDevice ) );
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice ) );
        if (!prop.deviceOverlap) {
            printf( "Device will not handle overlaps, so no speed up from streams\n" );
            return;
        }

        /* allocate data */
        // gIntensity_h_ = (float*)malloc(gNumObjects * sizeof(float));
        HANDLE_ERROR(cudaHostAlloc((void**)&gIntensity_h_, gNumObjects * sizeof(float*), cudaHostAllocDefault));
        #if NORMAL_GPU
        HANDLE_ERROR(cudaMalloc((void**)&gIntensityIn_d_, gNumObjects * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void**)&gIntensityOut_d_, gNumObjects * sizeof(float)));
        #else
        #endif
        printf("--> Intensity memory has been allocated.\n");

        /* initialize the first iteration temperature */
        for (int i = 0; i < gNumObjects; i++) {
            gIntensity_h_[i] = 0;
        }

        for (int i = 0; i < HEAT_SOURCE_SIZE; i++) {
            gIntensity_h_[heatSource[i]] = 1.0;
        }
        // HANDLE_ERROR(cudaMemcpy(gIntensityIn_d_, gIntensity_h_, sizeof(float) * gNumObjects, cudaMemcpyHostToDevice));

        printf("--> Heat source IDs have been determined.\n");
        printf("--> Initialization done.\n");
        printf("--> propagating start...\n");
    }
    else {
        TIMING_BEGIN
        bvhInstance->propagate();
        TIMING_END("time cost:")
    }
}


void quit_heatTransfer() {
    // free(gIntensity_h_);
    HANDLE_ERROR(cudaFree(gIntensity_h_));
    #if NORMAL_GPU
    HANDLE_ERROR(cudaFree(gIntensityIn_d_));
    HANDLE_ERROR(cudaFree(gIntensityOut_d_));
    #else
    #endif
}
