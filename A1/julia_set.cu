/**
 * @file julia_set.cu
 * @author HangX-Ma (m-contour@qq.com)
 * @brief julia set realization with gradient colors
 * @version 0.1
 * @date 2022-09-22
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
#include "book.h"
#include "cpu_bitmap.h"

#define DIM 1000
#define FINITE_THRSHOLED    1000
#define ITERATION_THRESHOLD 260
#define MAX_COLOR           256


struct cuComplex {
    float r;
    float i;
    __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    /* rewrite cuComplex addition and multiplication operation */
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};


__device__ int julia( int x, int y, const float scale) {
    float px = scale * (float)(DIM/2 - x) / (DIM/2);
    float py = scale * (float)(DIM/2 - y) / (DIM/2);

    cuComplex c(0.33, 0.4);
    // cuComplex c(-0.8, 0.156);

    cuComplex a(px, py);

    int i;
    for (i = 0; i < ITERATION_THRESHOLD; i++) {
        a = a * a + c;
        if (a.magnitude2() > FINITE_THRSHOLED) {
            break;
        }
    }
    return i;
}


inline __device__ float dev_fabs(float num) {
    return fabs(num);
}

inline __device__ float dev_fmod(float num, float mod) {
    return fmod(num, mod);
}

__global__ void kernel( unsigned char *ptr, int* dev_Pallette) {
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    int julia_iter = julia( x, y , 1.5);

    int color_sel = julia_iter % MAX_COLOR;
    ptr[offset*4 + 0] = dev_Pallette[color_sel*3];
    ptr[offset*4 + 1] = dev_Pallette[color_sel*3+1];
    ptr[offset*4 + 2] = dev_Pallette[color_sel*3+2];
    ptr[offset*4 + 3] = 255;
}

void HSL2RGB(float h, float s, float l, int* ret)
{
    const float C = (1 - fabs(2 * l - 1)) * s; // chroma
    const float H = h / 60;
    const float X = C * (1 - fabs(fmod(H, 2) - 1));
    float rgb[3] = {0};

    if (H > 0 && H < 1)  
        rgb[0] = C, rgb[1] = X, rgb[2] = 0;
    else if (H >= 1 && H < 2) 
        rgb[0] = X, rgb[1] = C, rgb[2] = 0;
    else if (H >= 2 && H < 3) 
        rgb[0] = 0, rgb[1] = C, rgb[2] = X;
    else if (H >= 3 && H < 4) 
        rgb[0] = 0, rgb[1] = X, rgb[2] = C;
    else if (H >= 4 && H < 5) 
        rgb[0] = X, rgb[1] = 0, rgb[2] = C;
    else if (H >= 5 && H < 6) 
        rgb[0] = C, rgb[1] = 0, rgb[2] = X;
    else                      
        rgb[0] = 0, rgb[1] = 0, rgb[2] = 0;

    const float m = l - 0.5 * C;

    *ret     = (int)(rgb[0] + m) * 255;
    *(ret+1) = (int)(rgb[1] + m) * 255;
    *(ret+2) = (int)(rgb[2] + m) * 255;
}


void InitColorSet(float h1 /* = 137.0 */, float h2 /* = 30.0 */, int* Pallette) {
    for (int i = 0; i < MAX_COLOR / 2; ++i) {
        HSL2RGB(h1, 20.0, i, Pallette + i * 3);
        HSL2RGB(h2, 30.0, i, Pallette + (MAX_COLOR - 1 - i) * 3);
    }
}


// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock      data;
    CPUBitmap      bitmap( DIM, DIM, &data );
    unsigned char *dev_bitmap;

    int Pallette[MAX_COLOR*3] = {0};
    int* dev_Pallette;

    InitColorSet(260.0, 60.0, Pallette);

    HANDLE_ERROR( cudaMalloc( (void**)&dev_Pallette, MAX_COLOR*3 ) );
    HANDLE_ERROR( cudaMemcpy( dev_Pallette, Pallette,
                              MAX_COLOR*3,
                              cudaMemcpyHostToDevice) );

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
    data.dev_bitmap = dev_bitmap;

    dim3    grid(DIM,DIM);
    kernel<<<grid,1>>>( dev_bitmap, dev_Pallette);

    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );
                              
    HANDLE_ERROR( cudaFree( dev_Pallette ) );
    HANDLE_ERROR( cudaFree( dev_bitmap ) );

    bitmap.display_and_exit();
}