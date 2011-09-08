#include "header.h"

#include <stdio.h>
#include <cuda.h>

#include <iostream>

// Kernel that executes on the CUDA device
__global__ void square_array(float *a, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx<N) a[idx] = a[idx] * a[idx];
}

void caller() {
	cudaSetDevice(1);
	
	float *a_h;
	const int N = 10;  // number of elements
	
	size_t size = N * sizeof(float);
	cudaMallocHost(&a_h, size);
	
	for (int i=0; i<N; i++)
		a_h[i] = i;
	
	int block_size = 4;
	int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	
	square_array <<< n_blocks, block_size >>> (a_h, N);
	
	cudaDeviceSynchronize();
	
	for (int i=0; i<N; i++) printf("%d %f\n", i, a_h[i]);
	
	cudaFreeHost(a_h);
}
