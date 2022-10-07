/**
 * @file dynamic_ray_tracer.cu
 * @author HangX-Ma (m-contour@qq.com)
 * @brief dynamic ray tracer based on CUDA acceleration
 * @version 0.1
 * @date 2022-10-05
 * 
 * @copyright Copyright (c) 2022 HangX-Ma(MContour) m-contour@qq.com
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda.h"
#include "book.h"
#include "cpu_bitmap.h"
#include "cpu_anim.h"
#include "helper_math.h"

#define SPHERES 10
#define DIM (1024)
#define EDGE_DIM (500)
#define THREAD_NUM (32)
#define MAX_VEL (20.0f)
#define MAX_RADIUS (70)
#define MIN_RADIUS (30)
#define INF 2e10f
#define rnd(x) ((float)x*rand()/RAND_MAX)

// #define NORMAL 1
// #define RUN_THROUGH 1
#define REBOUND 1

struct Sphere {
    float3 RGB;
    float3 trans;
    float3 vel;
    float radius;
    bool ifReverseVelDir;

    __host__ void init() {
        RGB = make_float3(rnd(1.0f), rnd(1.0f), rnd(1.0f));
        trans = make_float3(rnd(EDGE_DIM*2) - EDGE_DIM, rnd(EDGE_DIM*2) - EDGE_DIM, rnd(EDGE_DIM*2) - EDGE_DIM);
        vel = make_float3(rnd(MAX_VEL) - MAX_VEL / 2, rnd(MAX_VEL) - MAX_VEL / 2, 0);
        radius = rnd(40.0f) + MIN_RADIUS;
    }

    __device__ float hit(float2 cam_trans, float3 *n, int ticks) {
        float3 curr_trans;

        #if NORMAL
        curr_trans = trans + vel * ticks;
        #else
        float x, y;
        char2 ifReverse = make_char2(0, 0);
        float3 tmp_trans = trans + vel * ticks;

        /* check if x or y is negtive or not.
         Reverse to positive number if x or y is negtive */
        if (tmp_trans.x < 0) {
            tmp_trans.x = -tmp_trans.x;
            ifReverse.x = 1;
        }

        if (tmp_trans.y < 0) {
            tmp_trans.y = -tmp_trans.y;
            ifReverse.y = 1;
        }

        x = tmp_trans.x + EDGE_DIM;
        y = tmp_trans.y + EDGE_DIM;

        #if RUN_THROUGH
        curr_trans = make_float3(((int)x % DIM) - EDGE_DIM, 
                                 ((int)y % DIM) - EDGE_DIM, 
                                 0);
        #endif

        #if REBOUND
        if ((int)(x / DIM) % 2 == 0) {
            x = (int)x % DIM - EDGE_DIM;
        } else {
            x = DIM - (int)x % DIM - EDGE_DIM;
        }

        if ((int)(y / DIM) % 2 == 0) {
            y = (int)y % DIM - EDGE_DIM;
        } else {
            y = DIM - (int)y % DIM - EDGE_DIM;
        }

        if (ifReverse.x == 1) x = -x;
        if (ifReverse.y == 1) y = -y;

        curr_trans = make_float3(x, y, 0);
        #endif

        #endif

        float dist = length(make_float2(cam_trans.x - curr_trans.x, cam_trans.y - curr_trans.y));
        if (dist < radius) {
            float3 tmp_RGB;
            float dz = sqrtf(radius * radius - dist * dist);
            tmp_RGB = RGB * 255 * dz / radius + ticks * MAX_VEL;
            *n = make_float3((int)tmp_RGB.x % 255, (int)tmp_RGB.y % 255, (int)tmp_RGB.z % 255);
            return curr_trans.z + dz;
        }

        return -INF;
    }
};

__constant__ Sphere dev_s[SPHERES];

__global__ void kernel(unsigned char *ptr, int ticks) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float ox = (x - DIM/2);
    float oy = (y - DIM/2);

    float3 RGB = make_float3(0, 0, 0);
    float2 cam_trans = make_float2(ox, oy);
    float   maxz = -INF;
    for(int i = 0; i < SPHERES; i++) {
        float3 n;
        float z = dev_s[i].hit(cam_trans, &n, ticks);
        if (z > maxz) {
            RGB = n;
            maxz = z;
        }
    }

    ptr[offset*4 + 0] = (int)(RGB.x);
    ptr[offset*4 + 1] = (int)(RGB.y);
    ptr[offset*4 + 2] = (int)(RGB.z);
    ptr[offset*4 + 3] = 255;
}


__global__ void radius_kernel() {
    int seed = threadIdx.x % SPHERES;
    float newRadius = 1.1 * dev_s[seed].radius;
    float newRadius_sec = newRadius - MAX_RADIUS;
    dev_s[seed].radius = newRadius > MAX_RADIUS ? 
                        (newRadius_sec < MIN_RADIUS ? newRadius_sec + MIN_RADIUS : newRadius_sec) 
                        : newRadius;
}


// globals needed by the update routine
struct DataBlock {
    unsigned char *dev_bitmap;
    CPUAnimBitmap *bitmap;
    cudaEvent_t start, stop;
    float totalTime;
    float frames; 
};


void anim_gpu(DataBlock *d, int ticks) {
    /* time record: START */
    HANDLE_ERROR(cudaEventRecord(d->start, 0));

    /* FUNCTIONAL CODE HERE */
    dim3 blocks(DIM/THREAD_NUM,DIM/THREAD_NUM);
    dim3 threads(THREAD_NUM,THREAD_NUM);
    CPUAnimBitmap *bitmap = d->bitmap;

    /* change radius */
    radius_kernel<<<blocks, threads>>>();
    /* ray tracing */
    kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);
    // copy our bitmap back from the GPU for display
    HANDLE_ERROR(cudaMemcpy(bitmap->get_ptr(), 
                            d->dev_bitmap,
                            bitmap->image_size(),
                            cudaMemcpyDeviceToHost));

    /* time record: END */
    HANDLE_ERROR(cudaEventRecord(d->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(d->stop));
    float elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, d->start, d->stop ) );
    d->totalTime += elapsedTime;
    ++d->frames;
    printf( "Average Time per frame: %3.1f ms\n", d->totalTime/d->frames ); 
}


void anim_exit(DataBlock *d) {
	HANDLE_ERROR(cudaFree(d->dev_bitmap));
    HANDLE_ERROR(cudaEventDestroy(d->start));
    HANDLE_ERROR(cudaEventDestroy(d->stop)); 
}


int main( void ) {
    DataBlock data;
    CPUAnimBitmap bitmap( DIM, DIM, &data );

    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;

    /* Initilize the data buffer */
    HANDLE_ERROR(cudaEventCreate(&data.start));
    HANDLE_ERROR(cudaEventCreate(&data.stop));

    HANDLE_ERROR(cudaMalloc((void**)&data.dev_bitmap, bitmap.image_size()));

    Sphere *temp_s = (Sphere*)malloc( sizeof(Sphere) * SPHERES );
    for(int i = 0; i < SPHERES; i++) {
        temp_s[i].init();
    }

    HANDLE_ERROR(cudaMemcpyToSymbol(dev_s, temp_s, sizeof(Sphere) * SPHERES));
    free(temp_s);

    bitmap.anim_and_exit((void(*)(void*,int))anim_gpu, 
                            (void(*)(void*))anim_exit); 
}