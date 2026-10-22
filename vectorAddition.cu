
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define cudaCheckError(ans) { gpuAssert((ans),__FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d \n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#define gpuKernelCheck() { gpuKernelAssert(__FILE__, __LINE__);}
inline void gpuKernelAssert(const char* file, int line, bool abort = true) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s %s %d \n", cudaGetErrorString(err), file, line);
		if (abort) exit(err);
	}
}

// CUDA Kernel for vector addition
__global__ void vectorAdd(int* A, int* B, int* C, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n) {
		C[i] = A[i] + B[i];
	}

}


int main() {
	int* A, * B, * C;
	int* d_A, * d_B, * d_C;
	long long SIZE = 1024 * 1024 * 1024 * 20;
	long size = SIZE * sizeof(int);

	// CUDA event creation, used for timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaError_t err;

	// Allocate device vectors
	cudaCheckError(cudaMalloc((void**)&d_A, size));

	err = cudaMalloc((void**)&d_B, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory - %s \n", cudaGetErrorString(err));
	}
	err = cudaMalloc((void**)&d_C, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory - %s \n", cudaGetErrorString(err));
	}

	A = (int*)malloc(size);
	B = (int*)malloc(size);
	C = (int*)malloc(size);
	for (int i = 0; i < SIZE; i++) {
		A[i] = i;
		B[i] = SIZE - i;
	}
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	int threadsPerBlock = 48;
	int blocksPerGrid = (SIZE * threadsPerBlock - 1) / threadsPerBlock;
	vectorAdd << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, SIZE);

	gpuKernelCheck();

	cudaEventRecord(stop);
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Execution time: %f milliseconds\n", milliseconds);
	cudaEventSynchronize(stop);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(A);
	free(B);
	free(C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}
