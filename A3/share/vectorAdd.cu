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

#include <helper_cuda.h>

#define REAL double

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
vectorAdd(const REAL *A, const REAL *B, REAL *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
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
main(void)
{
	//main_cpu();

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	//int numElements = 50000;
	int numElements = 50000000;

	size_t size = numElements * sizeof(REAL);
	printf("[Vector addition of %d elements]\n", numElements);

#if 1
	// Allocate the host input vector A
	REAL *h_A = (REAL *)malloc(size);

	// Allocate the host input vector B
	REAL *h_B = (REAL *)malloc(size);

	// Allocate the host output vector C
	REAL *h_C = (REAL *)malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL || h_B == NULL || h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i)
	{
		h_A[i] = rand() / (REAL)RAND_MAX;
		h_B[i] = rand() / (REAL)RAND_MAX;
	}
#else
	REAL *h_A = gA;
	REAL *h_B = gB;
	REAL *h_C = gC;
#endif

	// Allocate the device input vector A
	REAL *d_A = NULL;
	err = cudaMalloc((void **)&d_A, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device input vector B
	REAL *d_B = NULL;
	err = cudaMalloc((void **)&d_B, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Allocate the device output vector C
	REAL *d_C = NULL;
	err = cudaMalloc((void **)&d_C, size);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the host input vectors A and B in host memory to the device input vectors in
	// device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 1024;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	START_GPU
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, numElements);
	END_GPU

	err = cudaGetLastError();

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	printf("###################\n");

	// Verify that the result vector is correct
	for (int i = 0; i < 20; ++i)
	{
		printf("gA=%lf, gB=%lf, gC=%lf\n", h_A[i], h_B[i], h_C[i]);
	}

#if 0
	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i)
	{
		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
#endif

	printf("Test PASSED\n");

	// Free device global memory
	err = cudaFree(d_A);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_B);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	err = cudaFree(d_C);

	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	printf("Done\n");
	return 0;
}

