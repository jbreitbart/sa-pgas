#include "header.h"

#include <stdio.h>
#include <cuda.h>
#include <omp.h>
#include <iostream>
#include <cstdlib>

#include "adabs/vector.h"
#include "adabs/cuda_device/accessors.h"

__device__ void down (int *temp) {
  int offset = 1;

  for (int d = (blockDim.x*2)>>1; d>0; d >>= 1) {
    __syncthreads();
    if (threadIdx.x < d) {
      const int ai = offset * (2*threadIdx.x+1) - 1;
      const int bi = offset * (2*threadIdx.x+2) - 1;
      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
}

__device__ void up (int *temp) {
  int offset = blockDim.x*2;
  
  for (int d=1; d<(blockDim.x*2); d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (threadIdx.x < d) {
      const int ai = offset * (2*threadIdx.x+1) - 1;
      const int bi = offset * (2*threadIdx.x+2) - 1;
      const int t = temp[ai];
      temp[ai]  = temp[bi];
      temp[bi] += t;
    }
  }
}

__global__ void block_scan (int* input, int* output, int* block_sum) {
	volatile const int* reader = get_tile(input, blockIdx.x, 0, blockDim.x*2);
	
	__shared__ int temp[256];
	temp[2*threadIdx.x]   = reader[2*threadIdx.x];
	temp[2*threadIdx.x+1] = reader[2*threadIdx.x+1];
	
	down(temp);
	__syncthreads();
	
	int* block_sum_writer = get_tile_unitialized(block_sum, blockIdx.x, 0, 1);
	if (threadIdx.x == 0) {
		block_sum_writer[0] = temp[255];
		temp[255] = 0;
	}
	
	__syncthreads();
	set_tile(block_sum, blockIdx.x, 0, 1);
	
	up(temp);
	
	int* writer = get_tile_unitialized(output, blockIdx.x, 0, blockDim.x*2);
	writer[2*threadIdx.x]   = temp[2*threadIdx.x];
	writer[2*threadIdx.x+1] = temp[2*threadIdx.x+1];
	set_tile(output, blockIdx.x, 0, blockDim.x*2);
}


__global__ void block_add (int* input, int* output, int* block_sum_scanned) {
	volatile const int* reader = get_tile(input, blockIdx.x+1, 0, blockDim.x*2);
	volatile const int* block_sum_reader = get_tile(block_sum_scanned, blockIdx.x+1, 0, 1);
	
	__shared__ int temp[256];
	temp[2*threadIdx.x]   = reader[2*threadIdx.x];
	temp[2*threadIdx.x+1] = reader[2*threadIdx.x+1];
	
	int add = block_sum_reader[0];
	
	temp[2*threadIdx.x]   += add;
	temp[2*threadIdx.x+1] += add;
	
	int* writer = get_tile_unitialized(output, blockIdx.x+1, 0, blockDim.x*2);
	writer[2*threadIdx.x]   = temp[2*threadIdx.x];
	writer[2*threadIdx.x+1] = temp[2*threadIdx.x+1];
	set_tile(output, blockIdx.x+1, 0, blockDim.x*2);
}


void caller() {
    
    const int block_size = 256;
    const int nb_of_blocks = 200;
	adabs::vector< adabs::cuda_host::local <int> > input(nb_of_blocks*block_size, block_size);
	adabs::vector< adabs::cuda_host::local <int> > output_block_scan(nb_of_blocks*block_size, block_size);
	adabs::vector< adabs::cuda_host::local <int> > output_final(nb_of_blocks*block_size, block_size);
	
	adabs::vector< adabs::cuda_host::local <int> > block_sums(nb_of_blocks, 1);
	adabs::vector< adabs::cuda_host::local <int> > block_sums_scanned(nb_of_blocks, 1);
	
	cudaSetDevice(0);
	// start block scan kernel @ GPU 0
	block_scan<<<nb_of_blocks, block_size/2>>>((int*)input.get_distri().get_data_addr().get_raw_pointer(),
	                                           (int*)output_block_scan.get_distri().get_data_addr().get_raw_pointer(),
	                                           (int*)block_sums.get_distri().get_data_addr().get_raw_pointer()
	                                          );
	
	cudaSetDevice(1);
	// start add kernel @ GPU 1
	block_add<<<nb_of_blocks-1, block_size/2>>>((int*)output_block_scan.get_distri().get_data_addr().get_raw_pointer(),
	                                            (int*)output_final.get_distri().get_data_addr().get_raw_pointer(),
	                                            (int*)block_sums_scanned.get_distri().get_data_addr().get_raw_pointer()
	                                           );
	
	#pragma omp parallel numthread(4)
	{
		int me = omp_get_thread_num();
		int all = omp_get_num_threads();
		
		// fill input array with random numbers
		#pragma omp single nowait
		{
			for (int i=0; i<nb_of_blocks; ++i) {
				int *writer = input.get_unitialized(i*block_size);
				for (int j=0; j<block_size; ++j) {
					writer[j] = std::rand() % 256;//block_size - j;
				}
				input.set(i*block_size, writer);
			}
		}
	
		// scan block sums
		#pragma omp single nowait
		{
			int start = 0;
			for (int i=0; i<nb_of_blocks; ++i) {
				int *writer = block_sums_scanned.get_unitialized(i);
				writer[0] = start;
				block_sums_scanned.set(i, writer);
				
				const int reader = block_sums.get(i);
				start += reader;
			}
		}
		
		// copy block 0
		#pragma omp single nowait
		{
			const int *reader = output_block_scan.get_tile(0);
			int *writer = output_final.get_unitialized(0);
			for (int j=0; j<block_size; ++j) {
				writer[j] = reader[j];
			}
			output_final.set(0, writer);
		}
	
		// read final results + sanity check (may crash due to overflow)
		#pragma omp single nowait
		{
			int prev = -1;
			for (int i=0; i<nb_of_blocks; ++i) {
				const int *reader = output_final.get_tile(i*block_size);
				const int *reader2 = input.get_tile(i*block_size);
				const int *reader3 = output_block_scan.get_tile(i*block_size);
				const int reader4 = block_sums_scanned.get(i);
				
				for (int j=0; j<block_size; ++j) {
					if (prev > reader[j]) {
						for (int j=0; j<block_size; ++j) {
							std::cout << i*block_size+j << " - " << reader2[j] << " - " << reader3[j] << " - " << reader4 << " - " << reader[j] << std::endl;
						}
					}
					assert (prev <= reader[j]);
					prev = reader[j];
				}
			}
		}
	}
	
	// make sure our next call to cudaGetLastError will return errors
	// from the kernels started before
	cudaDeviceSynchronize();
	
	// check for error
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	
}

#if 0
__global__ void test(float *A, float *B) {
	float* writer = get_tile_unitialized(A, blockIdx.x, blockIdx.y, blockDim.x*blockDim.y);
	volatile const float* reader = get_tile(B, blockIdx.x, blockIdx.y, blockDim.x*blockDim.y);
	
	writer[threadIdx.x+threadIdx.y*blockDim.x] = reader[threadIdx.x+threadIdx.y*blockDim.x];
	__syncthreads();
	 
	set_tile(A, blockIdx.x, blockIdx.y, blockDim.x*blockDim.y);
}
#endif
	#if 0
void caller(/*adabs::matrix< adabs::cuda_host::local < float > > &A,
            adabs::matrix< adabs::cuda_host::local < float > > &B,
            adabs::matrix< adabs::cuda_host::local < float > > &C*/) {
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
	
	#endif

