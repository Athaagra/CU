#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <cuda.h>
//#define SIZE 2024


//Erro checking macro
#define cudaCheckError(ans) { gpuAssert((ans),__FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
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

int main(int argc, char **argv) {
	if (argc != 2) {
		fprintf(stderr, "Usage: %s <block size>\n", argv[0]);
		return 1;
	}
	// Parse block size from command line
	int threadsPerBlock = atoi(argv[1]);
	if (threadsPerBlock <= 0) {
		fprintf(stderr, "Block size must be a positive integer.\n");
		return 1;
	}
	int* A, * B, * C;
	int *d_A, *d_B, *d_C;
	long long SIZE=1024LL * 1024 * 32;
	long size = SIZE * sizeof(int);

	// CUDA event creation, used for //timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaError_t err;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	cudaCheckError(cudaMalloc((void **)&d_A, size));
	//Allocate and intialize host vectors
	// Alocate device vectors 
	cudaMalloc((void **)&d_A, size);
	
	//if (err != cudaSuccess){
	//fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
	//}
	err = cudaMalloc((void **)&d_B, size);
	if (err != cudaSuccess){
	fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc((void**)&d_C, size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
	}

	A = (int*)malloc(size);
	B = (int*)malloc(size);
	C = (int*)malloc(size);
	for (int i = 0; i < SIZE; i++) {
		A[i] = i;
		B[i] = SIZE - i;
	}

	//cudaMalloc((void**)&d_A, size);
	//cudaMalloc((void**)&d_B, size);
	//cudaMalloc((void**)&d_C, size);



//	cudaMalloc((void **)&d_C, size);
//	if (err != cudaSuccess){
//	fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
//}
	//Copy host vectors to device
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//Start recording
	cudaEventRecord(start);
	// Launch the Vector Add CUDA Kernel
	//int threadsPerBlock = 96;
	int blocksPerGrid = (SIZE * threadsPerBlock -1) / threadsPerBlock;
	vectorAdd <<<blocksPerGrid, threadsPerBlock >>> (d_A, d_B, d_C, SIZE);
	gpuKernelCheck();
	cudaEventRecord(stop);
	//Stop recording
	//cudaEventRecord(start);

	//Copy result back to host
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	//Calculate and print the execution time
	cudaEventSynchronize(stop);
	//float milliseconds = 0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("Execution time: %f milliseconds\n",milliseconds);

	//printf("\n Execution finished \n");
	//for (int i = 0; i < SIZE; i++) {
	//	printf("%d + %d = %d", A[i], B[i], C[i]);
	//	printf("\n");
	//}

	//Cleanup
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(A);
	free(B);
	free(C);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
