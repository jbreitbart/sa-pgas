#include "header.h"

#include <stdio.h>
#include <cuda.h>

#include <iostream>
#include "adabs/tools/alignment.h"


template<typename T>
__device__ volatile const T* get_tile(T* ptr, const int x, const int y, const int block_size) {
	int a = adabs::tools::alignment<T>::val();
	if (a<sizeof(int)) a = sizeof(int);
	
	char* tmp = (char*) ptr;
	tmp += a;
	tmp += (a + sizeof(T)*block_size)*(y*gridDim.x + x);
	
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		volatile int *flag = (volatile int*)(tmp - a);
		while (*flag != 3) {}
	}
	
	__syncthreads();
	return (T*)tmp;
}

template<typename T>
__device__ T* get_tile_unitialized(T* ptr, const int x, const int y, const int block_size) {
	int a = adabs::tools::alignment<T>::val();
	if (a<sizeof(int)) a = sizeof(int);
	
	char* tmp = (char*) ptr;
	tmp += a;
	tmp += (a + sizeof(T)*block_size)*(y*gridDim.x + x);
	
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		volatile int *flag = (volatile int*)(tmp - a);
		*flag = 1;
	}
	__threadfence_system();
	
	__syncthreads();
	return (T*)tmp;
}

template<typename T>
__device__ void set_tile(T* ptr, const int x, const int y, const int block_size) {
	int a = adabs::tools::alignment<T>::val();
	if (a<sizeof(int)) a = sizeof(int);
	
	char* tmp = (char*) ptr;
	tmp += a;
	tmp += (a + sizeof(T)*block_size)*(y*gridDim.x + x);
	
	__threadfence_system();
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		volatile int *flag = (volatile int*)(tmp - a);
		*flag = 3;
	}
	__threadfence_system();
	
	__syncthreads();
}

__global__ void test(float *A, float *B) {
	float* writer = get_tile_unitialized(A, blockIdx.x, blockIdx.y, blockDim.x*blockDim.y);
	volatile const float* reader = get_tile(B, blockIdx.x, blockIdx.y, blockDim.x*blockDim.y);
	
	writer[threadIdx.x+threadIdx.y*blockDim.x] = reader[threadIdx.x+threadIdx.y*blockDim.x];
	__syncthreads();
	 
	set_tile(A, blockIdx.x, blockIdx.y, blockDim.x*blockDim.y);
}

void caller(adabs::matrix< adabs::cuda_host::local < float > > &A,
            adabs::matrix< adabs::cuda_host::local < float > > &B,
            adabs::matrix< adabs::cuda_host::local < float > > &C) {
	cudaSetDevice(1);
	
	dim3 block_d(A.get_distri().get_batch_size_x(), A.get_distri().get_batch_size_y());
	dim3 grid_d(A.get_distri().get_size_x() / block_d.x, A.get_distri().get_size_y() / block_d.y);
	
	test <<< grid_d, block_d >>> ((float*)A.get_distri().get_data_addr().get_raw_pointer(), (float*)B.get_distri().get_data_addr().get_raw_pointer());

	for (int i=0; i<grid_d.x; ++i) {
		for (int j=0; j<grid_d.y; ++j) {
			float *b_ptr = B.get_tile_unitialized(i*block_d.x, j*block_d.y);
			for (int ii=0; ii<block_d.y; ++ii) {
				for (int jj=0; jj<block_d.x; ++jj) {
					b_ptr[ii*block_d.x + jj] = jj; 
				}
			}
			B.set_tile(i*block_d.x, j*block_d.y, b_ptr);
		}
	}

	cudaDeviceSynchronize();
	
	for (int i=0; i<grid_d.x; ++i) {
		for (int j=0; j<grid_d.y; ++j) {
			const float *a_ptr = A.get_tile(i*block_d.x, j*block_d.y);
			for (int ii=0; ii<block_d.y; ++ii) {
				for (int jj=0; jj<block_d.x; ++jj) {
					if (a_ptr[ii*block_d.x + jj] != jj) {
						for (int x = jj; x<jj+10 && x<block_d.x; ++x)
							std::cout << "(" << i << ", " << j << ", " << ii << ", " << x << ") - " << a_ptr[ii*block_d.x + x] << std::endl; 
					}
					assert (a_ptr[ii*block_d.x + jj] == jj);
				}
			}
		}
	}
	

	// check for error
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	
}
