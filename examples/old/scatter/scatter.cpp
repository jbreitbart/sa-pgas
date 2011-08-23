#include <iostream>
#include <cmath>
#include <omp.h>
#include <pthread.h>

#include "adabs/adabs.h"

#include "adabs/matrix.h"
#include "adabs/remote_matrix.h"
#include "adabs/pgas_ref.h"

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
	using adabs::me;
	using adabs::all;

	timeval tv1, tv2;

	adabs::init(&argc, &argv);

	omp_set_num_threads(THREADS);
	
	std::cout << "hello from " << me << " of " << all << std::endl;
	adabs::barrier_wait();
	
	typedef adabs::distributed::matrix<double, TILE_SIZE, adabs::distributed::row_distribution<THREADS> > matrixT;
	
	matrixT dist_matrix(SIZE, SIZE);
	dist_matrix.use();
	
	adabs::matrix<double, TILE_SIZE> local_matrix(SIZE, SIZE);
	init_matrix(local_matrix, 0, 1);
	
	std::cout << me << ": done creating" << std::endl;
	adabs::barrier_wait();

	adabs::collective::vector < adabs::pgas_ref<volatile int> > ex(all);
	 
	volatile int i=me;
	
	ex.set(me, adabs::pgas_ref<volatile int>(me, &i));
	adabs::barrier_wait();
	
	adabs::pgas_ref<volatile int> remote_ref = const_cast< adabs::pgas_ref<volatile int>& > (ex.get(adabs::next));
	
	remote_ref = me;
	
	adabs::barrier_wait();
	std::cout << me << ": i = " << i << std::endl;
	adabs::barrier_wait();


	if (me == 0) {
		std::cout << me << ": start scatter" << std::endl;
		dist_matrix = local_matrix;
		std::cout << me << ": end scatter" << std::endl;
	}
	
	adabs::exit(0);

	return 0;
} 

