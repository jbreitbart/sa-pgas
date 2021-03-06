#include <iostream>
#include <vector>
#include <omp.h>
#include <sys/time.h>

#include "adabs/adabs.h"
#include "adabs/cuda_host/allocator.h"
#include "adabs/cuda_host/pgas_addr.h"

#include "adabs/cuda_host/local.h"
#include "adabs/matrix.h"

#include "header.h"


using namespace std;

const int THREADS = 4;

int main(int argc, char *argv[]) {
	using adabs::me;
	using adabs::all;
	using adabs::next;
	
	adabs::init(&argc, &argv);

	omp_set_num_threads(THREADS);
	
	/*adabs::matrix< adabs::cuda_host::local < float > > A(512, 512, 16);
	adabs::matrix< adabs::cuda_host::local < float > > B(512, 512, 16);
	adabs::matrix< adabs::cuda_host::local < float > > C(512, 512, 16);*/
	
	caller();
	
	adabs::cuda_host::pgas_addr<int> itile = adabs::cuda_host::allocator<int>::allocate(100*128, 128);
	
	#pragma omp parallel
	{
		if (omp_get_thread_num() == 0) {
			int * ptr = itile.get_data_unitialized();
			
			for (int i=0; i<128; ++i) {
				ptr[i] = i;
			}
			
			itile.set_data(ptr);
		}
		
		adabs::cuda_host::pgas_addr<int> tlocal = itile + omp_get_thread_num();
		
		for (int i=omp_get_thread_num()+1; i<100; i+=omp_get_num_threads()) {
			const int *old_ptr = tlocal.get_data();
			
			tlocal += 1;
			
			int *new_ptr = tlocal.get_data_unitialized();
			
			for (int i=0; i<128; ++i) {
				new_ptr[i] = old_ptr[i];
			}
			
			tlocal.set_data(new_ptr);
			
			tlocal += omp_get_num_threads()-1;
		}
		
		if (omp_get_thread_num() == THREADS-1) {
			tlocal -= 1;
			
			const int *ptr = tlocal.get_data();
			for (int i=0; i<128; ++i) {
				assert (i == ptr[i]); 
			}
		}
	}
	
	adabs::cuda_host::allocator<int>::deallocate(itile);
	
	adabs::barrier_wait();
	
	std::cout << me << ": " << "Everything fine!" << std::endl;

	adabs::exit(0);
} 
