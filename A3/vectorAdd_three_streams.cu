/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /**
  * Vector addition: C = A + B.
  *
  * This sample is a very basic sample that implements element by element
  * vector addition. It is the same as the sample illustrating Chapter 2
  * of the programming guide with some additions like error checking.
  */

#include <stdio.h>
  // For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "book.h"

#define REAL double

#define N (1024*1024UL)
#define FULL_DATA (N*10UL)

#define START_GPU {\
cudaEvent_t     start, stop;\
float   elapsedTime;\
checkCudaErrors(cudaEventCreate(&start)); \
checkCudaErrors(cudaEventCreate(&stop));\
checkCudaErrors(cudaEventRecord(start, 0));\

#define END_GPU \
checkCudaErrors(cudaEventRecord(stop, 0));\
checkCudaErrors(cudaEventSynchronize(stop));\
checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop)); \
printf("GPU Time used:  %3.1f ms\n", elapsedTime);\
checkCudaErrors(cudaEventDestroy(start));\
checkCudaErrors(cudaEventDestroy(stop));}

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const REAL *A, const REAL *B, REAL *C, size_t numElements)
{
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {
		//C[i] = A[i] + B[i];
		C[i] = atan(A[i]) / (fabs(sin(fabs(B[i]) + 0.0001)) + 0.1);
	}
}

extern double *gA, *gB, *gC;
int main_cpu();

/**
 * Host main routine
 */
int
main(void) {
    /* device property checker */
	cudaDeviceProp  prop;
    int whichDevice;
    HANDLE_ERROR( cudaGetDevice( &whichDevice ) );
    HANDLE_ERROR( cudaGetDeviceProperties( &prop, whichDevice ) );
    if (!prop.deviceOverlap) {
        printf( "Device will not handle overlaps, so no speed up from streams\n" );
        return 0;
    }

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	size_t N_size = N * sizeof(REAL);
	size_t data_size = FULL_DATA * sizeof(REAL);

	printf("[Vector addition of %ld elements]\n", FULL_DATA);


    cudaStream_t stream0, stream1, stream2;
    REAL *host_a, *host_b, *host_c;
    REAL *dev_a0, *dev_b0, *dev_c0;
    REAL *dev_a1, *dev_b1, *dev_c1;
    REAL *dev_a2, *dev_b2, *dev_c2;

    // initialize the streams
    HANDLE_ERROR( cudaStreamCreate( &stream0 ) );
    HANDLE_ERROR( cudaStreamCreate( &stream1 ) );
    HANDLE_ERROR( cudaStreamCreate( &stream2 ) );

    // allocate the memory on the GPU
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a0, N_size ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b0, N_size ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c0, N_size ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a1, N_size ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b1, N_size ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c1, N_size ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_a2, N_size ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_b2, N_size ) );
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c2, N_size ) );

    // allocate host locked memory, used to stream
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_a,
                              data_size,
                              cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_b,
                              data_size,
                              cudaHostAllocDefault ) );
    HANDLE_ERROR( cudaHostAlloc( (void**)&host_c,
                              data_size,
                              cudaHostAllocDefault ) );

	// Initialize the host input vectors
	for (size_t i = 0; i < FULL_DATA; ++i) {
		host_a[i] = rand() / (REAL)RAND_MAX;
		host_b[i] = rand() / (REAL)RAND_MAX;
	}


	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 1024;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	START_GPU
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // now loop over full data, in bite-sized chunks
    for (size_t i = 0; i < FULL_DATA; i += N*3) {
        // enqueue copies of a in stream0 and stream1
        HANDLE_ERROR( cudaMemcpyAsync( dev_a0, host_a+i,
                                       N_size,
                                       cudaMemcpyHostToDevice,
                                       stream0 ) );
        HANDLE_ERROR( cudaMemcpyAsync( dev_a1, host_a+i+N,
                                       N_size,
                                       cudaMemcpyHostToDevice,
                                       stream1 ) );
        HANDLE_ERROR( cudaMemcpyAsync( dev_a2, host_a+i+N*2,
                                       N_size,
                                       cudaMemcpyHostToDevice,
                                       stream2 ) );
        // enqueue copies of b in stream0 and stream1
        HANDLE_ERROR( cudaMemcpyAsync( dev_b0, host_b+i,
                                       N_size,
                                       cudaMemcpyHostToDevice,
                                       stream0 ) );
        HANDLE_ERROR( cudaMemcpyAsync( dev_b1, host_b+i+N,
                                       N_size,
                                       cudaMemcpyHostToDevice,
                                       stream1 ) );
        HANDLE_ERROR( cudaMemcpyAsync( dev_b2, host_a+i+N*2,
                                       N_size,
                                       cudaMemcpyHostToDevice,
                                       stream2 ) );
        // enqueue kernels in stream0 and stream1   
        vectorAdd<<<blocksPerGrid, threadsPerBlock,0,stream0>>>( dev_a0, dev_b0, dev_c0, N );
        vectorAdd<<<blocksPerGrid, threadsPerBlock,0,stream1>>>( dev_a1, dev_b1, dev_c1, N );
        vectorAdd<<<blocksPerGrid, threadsPerBlock,0,stream2>>>( dev_a2, dev_b2, dev_c2, N );

        // enqueue copies of c from device to locked memory
        HANDLE_ERROR( cudaMemcpyAsync( host_c+i, dev_c0,
                                       N_size,
                                       cudaMemcpyDeviceToHost,
                                       stream0 ) );
        HANDLE_ERROR( cudaMemcpyAsync( host_c+i+N, dev_c1,
                                       N_size,
                                       cudaMemcpyDeviceToHost,
                                       stream1 ) );
        HANDLE_ERROR( cudaMemcpyAsync( host_c+i+N*2, dev_c2,
                                       N_size,
                                       cudaMemcpyDeviceToHost,
                                       stream2 ) );
    }
    HANDLE_ERROR( cudaStreamSynchronize( stream0 ) );
    HANDLE_ERROR( cudaStreamSynchronize( stream1 ) );
    HANDLE_ERROR( cudaStreamSynchronize( stream2 ) );

	END_GPU

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("###################\n");
	// Verify that the result vector is correct
	for (size_t i = 0; i < 20; ++i) {
		printf("gA=%lf, gB=%lf, gC=%lf\n", host_a[i], host_b[i], host_c[i]);
	}

	// Verify that the result vector is correct
	for (size_t i = 0; i < FULL_DATA; ++i)
	{
		// if (fabs(host_a[i] + host_b[i] - host_c[i]) > 1e-5)
		// {
		// 	fprintf(stderr, "Result verification failed at element %ld!\n", i);
		// 	exit(EXIT_FAILURE);
		// }
		if (fabs(atan(host_a[i])/((fabs(sin(fabs(host_b[i]) + 0.0001)) + 0.1)) - host_c[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %ld!\n", i);
			exit(EXIT_FAILURE);
		}
	}
	printf("Test PASSED\n");

    // cleanup the streams and memory
    HANDLE_ERROR( cudaFreeHost( host_a ) );
    HANDLE_ERROR( cudaFreeHost( host_b ) );
    HANDLE_ERROR( cudaFreeHost( host_c ) );
    HANDLE_ERROR( cudaFree( dev_a0 ) );
    HANDLE_ERROR( cudaFree( dev_b0 ) );
    HANDLE_ERROR( cudaFree( dev_c0 ) );
    HANDLE_ERROR( cudaFree( dev_a1 ) );
    HANDLE_ERROR( cudaFree( dev_b1 ) );
    HANDLE_ERROR( cudaFree( dev_c1 ) );
    HANDLE_ERROR( cudaFree( dev_a2 ) );
    HANDLE_ERROR( cudaFree( dev_b2 ) );
    HANDLE_ERROR( cudaFree( dev_c2 ) );
    HANDLE_ERROR( cudaStreamDestroy( stream0 ) );
    HANDLE_ERROR( cudaStreamDestroy( stream1 ) );
    HANDLE_ERROR( cudaStreamDestroy( stream2 ) );

	printf("Done\n");

	return 0;
}

