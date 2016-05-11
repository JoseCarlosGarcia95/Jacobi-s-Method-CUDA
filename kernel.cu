
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define JACOBI_DEBUG 1
#define TOL 0.0001

__device__ int flag;

__global__ void internal_jacobi_solve(float * a, float * x0, float * b, unsigned int matrixSize) {
	unsigned int i, j;
	float sigma = 0, newValue;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	for (j = 0; j < matrixSize; j++) {
		if (i != j) {
			sigma = sigma + a[i*matrixSize + j] * x0[j];
		}
	}

	newValue = (b[i] - sigma) / a[i*matrixSize + i];

	if (abs(x0[i] - newValue) > TOL) flag = 0;
	x0[i] = newValue;
}
cudaError_t cuda_jacobi_solve(float ** a, float * x0, float * b, unsigned int matrixSize, int * iter) {
	unsigned int i, j;
	int  blockSize, minGridSize, gridSize, cpuConvergenceTest, k;
	float *extended_a = 0, *dev_a = 0, *dev_x0 = 0, *dev_b = 0, *dev_old = 0;

	cudaError_t cudaStatus;

	if (JACOBI_DEBUG) printf("Hello GPU0\n");
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	if (JACOBI_DEBUG) printf("Hello CPU\n");


	if (JACOBI_DEBUG) printf("Generating a vector with NxN size with a values\n");
	extended_a = (float*)malloc(matrixSize*matrixSize*sizeof(float));
	for (i = 0; i < matrixSize; i++) {
		for (j = 0; j < matrixSize; j++) {
			extended_a[i*matrixSize + j] = a[i][j];
		}
	}
	if (JACOBI_DEBUG) printf("Generated a vector with NxN size with a values\n");


	if (JACOBI_DEBUG) printf("Allocating memory in GPU\n");

	cudaStatus = cudaMalloc((void**)&dev_a, matrixSize*matrixSize*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_x0, matrixSize*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, matrixSize*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_old, matrixSize*sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	if (JACOBI_DEBUG) printf("Allocated memory in GPU\n");

	if (JACOBI_DEBUG) printf("Copying memory in GPU\n");

	cudaStatus = cudaMemcpy(dev_a, extended_a, matrixSize*matrixSize* sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_x0, x0, matrixSize* sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_old, x0, matrixSize* sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, matrixSize* sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cpuConvergenceTest = 0;
	k = 0;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, internal_jacobi_solve, 0, matrixSize);
	gridSize = (matrixSize + blockSize - 1) / blockSize;

	for (i = 0; i < 100 && !cpuConvergenceTest; i++) {
		cpuConvergenceTest = 1;
		cudaMemcpyToSymbol(flag, &cpuConvergenceTest, sizeof(int));


		internal_jacobi_solve << <gridSize, blockSize >> >(dev_a, dev_x0, dev_b, matrixSize);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching jacobi!\n", cudaStatus);
			goto Error;
		}

		cudaMemcpyFromSymbol(&cpuConvergenceTest, flag, sizeof(int));

		k++;
	}


	*iter = k;
	cudaStatus = cudaMemcpy(x0, dev_x0, matrixSize* sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	free(extended_a);
	cudaFree(dev_a);
	cudaFree(dev_x0);
	cudaFree(dev_b);
	return cudaStatus;
}

int main()
{
	unsigned int matrixSize = 1000, i, j;
	int iter;
	float *x0, *b, **test;
	// Allocate memory for CPU.
	test = (float**)malloc(sizeof(float*)*matrixSize);
	b = (float*)malloc(sizeof(float)*matrixSize);
	x0 = (float*)malloc(sizeof(float)*matrixSize);
	for (i = 0; i < matrixSize; i++) {
		test[i] = (float*)calloc(matrixSize, sizeof(float));

		// ONLY FOR TESTING PURPOSE.
		test[i][i] = 2;
		test[i][1] = 2*i;
		b[i] = 3;
		x0[i] = 0;
	}

	cuda_jacobi_solve(test, x0, b, matrixSize, &iter);


	printf("x0=(");
	for (i = 0; i < matrixSize; i++) {
		printf("%f,", x0[i]);
	}
	printf(")");

	system("pause");

	return 0;
}
