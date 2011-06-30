#include <iostream>
#include <cmath>
#include <omp.h>
#include <pthread.h>

#include "adabs/gasnet_config.h"

#include "adabs/adabs.h"
#include "adabs/matrix.h"
#include "adabs/remote_matrix.h"

#include "adabs/collective/vector.h"

#include "adabs/distributed/matrix.h"

using namespace std;

const int TILE_SIZE       = 64;
const int SIZE            = 4096;
const int THREADS         = 4;

/**
 * Fills the matrixes A with values.
 */
template<typename T>
static void init_matrix(T& A, const int start, const int jump) {
	#pragma omp parallel for
	for (int b=start; b<SIZE/TILE_SIZE; b+=jump) {
		for (int a=0; a<SIZE/TILE_SIZE; ++a) {
		
			double * restrict Aptr = A.get_tile_unitialized(a,b);
	
			for (int i=0; i<TILE_SIZE; ++i) {
				#pragma vector aligned
				for (int j=0; j<TILE_SIZE; ++j) {
					Aptr[i*TILE_SIZE+j] = i;
				}
			}
	
			A.set_tile(Aptr, a, b);
		
		}
	}
}


int main(int argc, char *argv[]) {
	timeval tv1, tv2;

	adabs::init(&argc, &argv);
	gasnet_set_waitmode(GASNET_WAIT_BLOCK);

	omp_set_num_threads(THREADS);
	
	const int me = gasnet_mynode();
	const int all = gasnet_nodes();

	std::cout << "hello from " << me << " of " << all << std::endl;
	
	adabs::barrier_wait();
	
	typedef adabs::distributed::matrix<double, TILE_SIZE, adabs::distributed::row_distribution<THREADS> > matrixT;
	
	matrixT dist_matrix(SIZE, SIZE);
	dist_matrix.use();
	
	adabs::matrix<double, TILE_SIZE> local_matrix(SIZE, SIZE);
	init_matrix(local_matrix, 0, 1);
	
	std::cout << me << ": done creating" << std::endl;
	adabs::barrier_wait();

	if (me == 0) {
		std::cout << me << ": start scatter" << std::endl;
		dist_matrix = local_matrix;
		std::cout << me << ": end scatter" << std::endl;
	}
	
	adabs::barrier_wait();
#if 0	
	// init all mas with data
	if (me==0) {
		for (int maI=0; maI<all+2; ++maI) {
			std::cout << me << ": " << maI << std::endl;
			for (int b=0; b<SIZE/TILE_SIZE; ++b) {
				for (int a=0; a<SIZE/TILE_SIZE; ++a) {
				
					double * restrict Aptr = mas[maI]->get_tile_unitialized(a,b);
					for (int i=0; i<TILE_SIZE; ++i) {
						#pragma vector aligned
						for (int j=0; j<TILE_SIZE; ++j) {
							Aptr[i*TILE_SIZE+j] = i;
						}
					}

					mas[maI]->set_tile(Aptr, a, b);
				}
			}
		}
	}
	
	adabs::barrier_wait();
	
	for (int maI=1; maI<all+2; ++maI) {
		mas[maI] -> enable_reuse(); 
	}
	
	for (int maI=1; maI<all+2; ++maI) {
		mas[maI] -> wait_for_reuse(); 
	}
	
	adabs::barrier_wait();
	
	for (int i=0; i<50; ++i) {
		matrixT &old = *mas[i%(all+2)];
		matrixT &cur = *mas[(i+1)%(all+2)];
		
		//std::cout << me << ": waiting for " << (i+1)%(all+2) << std::endl;
		cur.wait_for_reuse();
		cur.use();
		
		//std::cout << me << ": START " << i << " =================" << std::endl;
		
		compute_matrix(old, cur, me, all);
		
		//std::cout << me << ": END " << i << " =================" << std::endl;
		
		//std::cout << me << ": free " << (i)%(all+2) << std::endl;
		old.enable_reuse();
	}
	
	adabs::barrier_wait();
#endif
	return 0;
} 

