#pragma once

#ifdef __CUDACC__
#else
#error Include this file only with NVCC!
#endif

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

