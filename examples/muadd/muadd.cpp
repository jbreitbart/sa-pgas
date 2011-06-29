#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <pthread.h>
#include <sys/time.h>

#include "adabs/gasnet_config.h"

#include "adabs/adabs.h"
#include "adabs/matrix.h"
#include "adabs/remote_matrix.h"

#include "adabs/collective/vector.h"

#include "adabs/distributed/matrix.h"

using namespace std;

const int TILE_SIZE = 64;
const int SIZE = 4096*2;
const int THREADS = 4;




/**
 * Fills the matrixes A and B with values.
 */
template<typename T>
static void init_matrixes(T& A, T& B, const int start, const int jump) {
	#pragma omp parallel for
	for (int b=start; b<SIZE/TILE_SIZE; b+=jump) {
		for (int a=0; a<SIZE/TILE_SIZE; ++a) {
		
			double * restrict Aptr = A.get_tile_unitialized(a,b);
			double * restrict Bptr = B.get_tile_unitialized(a,b);
	
			for (int i=0; i<TILE_SIZE; ++i) {
				#pragma vector aligned
				for (int j=0; j<TILE_SIZE; ++j) {
					Aptr[i*TILE_SIZE+j] = i;
					Bptr[i*TILE_SIZE+j] = TILE_SIZE-i;
				}
			}
	
			A.set_tile(Aptr, a, b);
			B.set_tile(Bptr, a, b);
		
		}
	}
}

/**
 * Computes A*B=C for a single tile of C. The tile is defined by a, b.
 */
template<typename T>
static void multiply_matrix(const T& A, const T& B, T& C, const int a, const int b) {
	double * restrict Cptr  = C.get_tile_unitialized(a,b);
		  
	for (int i=0; i<TILE_SIZE; ++i) {
		#pragma vector aligned
		for (int j=0; j<TILE_SIZE; ++j) {
			Cptr[i*TILE_SIZE+j] = 0;
		}
	}

	for (int t=0; t<SIZE/TILE_SIZE; ++t) {
		const double * restrict Aptr = A.get_tile(a,t);
		const double * restrict Bptr = B.get_tile(t,b);

		for (int i=0; i<TILE_SIZE; ++i) {
			#pragma vector aligned
			for (int j=0; j<TILE_SIZE; ++j) {
				Cptr[i*TILE_SIZE+j] += Aptr[i*TILE_SIZE+j] * Bptr[i*TILE_SIZE+j]; 
			}
		}
	}

	C.set_tile(Cptr, a, b);				
}

/**
 *  Computes A+V=B for a single tile of B. The tile is defined by a, b.
 */
template<typename T, typename T2, typename T3>
static void add_vector(T& A, T2& B, const T3& V, const int a, const int b) {
	      double * restrict A1ptr = B.get_tile_unitialized(a,b);
	const double * restrict Aptr  = A.get_tile(a,b);
	
	for (int i=0; i<TILE_SIZE; ++i) {
		#pragma vector
		for (int j=0; j<TILE_SIZE; ++j) {
			A1ptr[i*TILE_SIZE+j] = Aptr[i*TILE_SIZE+j] + V[a*TILE_SIZE+i];
		}
	}
	
	
	B.set_tile(A1ptr, a, b);
}

/**
 *  Compares two tiles of B2 and B_seq.
 */
template<typename T1, typename T2>
static void compare(T1& B2, T2& B_seq, const int a, const int b) {
	const double * restrict Cptr = B2.get_tile(a,b);
	const double * restrict C_seqptr = B_seq.get_tile(a,b);
	
	for (int i=0; i<TILE_SIZE; ++i) {
		#pragma vector aligned
		for (int j=0; j<TILE_SIZE; ++j) {
			if (std::abs(Cptr[i*TILE_SIZE+j] - C_seqptr[i*TILE_SIZE+j]) > 0.0001) {
				std::cerr << Cptr[i*TILE_SIZE+j] << " - " << C_seqptr[i*TILE_SIZE+j] << std::endl;
			}
		}
	}
}

/**
 * Computes (A+V1) * (B+V2) = C with A, B, C being matrixes and V1, V2 vectors.
 * start and jump are the start / stride values used to allow a single process to only compute a fraction
 * of the whole computation.
 */
template <typename T, typename T2, typename T3>
static void compute(T3& A, T& B, T& C, T& A1, T& B2, T2& V1, T2& V2, const int start, const int jump) {
	#pragma omp parallel
	{
		
		#pragma omp for schedule(static) nowait
		for (int b=start; b<SIZE/TILE_SIZE; b+=jump) {
			for (int a=0; a<SIZE/TILE_SIZE; ++a) {
				add_vector(A, A1, V1, a, b);
			}
		}

		#pragma omp for schedule(static) nowait
		for (int b=start; b<SIZE/TILE_SIZE; b+=jump) {
			for (int a=0; a<SIZE/TILE_SIZE; ++a) {
				add_vector(B, B2, V2, a, b);
			}
		}
		
		#pragma omp for schedule(static) nowait
		for (int b=start; b<SIZE/TILE_SIZE; b+=jump) {
			// strided multiply, so different threads fetch different tiles
			for (int i=0; i<omp_get_num_threads(); ++i) {
				const int thread_start = (omp_get_thread_num()+i)%omp_get_num_threads();
				for (int a=thread_start; a<SIZE/TILE_SIZE; a+=omp_get_num_threads()) {
					multiply_matrix (A1, B2, C, a, b);
				}
			}
			/*for (int a=0; a<SIZE/TILE_SIZE; a++) {
				multiply_matrix (A1, B2, C, a, b);
			}*/
		}
	}
}

int main(int argc, char *argv[]) {
	timeval tv1, tv2;
	std::vector<double> V1(SIZE), V2(SIZE);
	for (int i=0; i<SIZE; ++i) {
		V1[i] = i;
		V2[i] = SIZE-i;
	}

	adabs::init(&argc, &argv);
	gasnet_set_waitmode(GASNET_WAIT_BLOCK);

	omp_set_num_threads(THREADS);
	
	const int me = gasnet_mynode();
	const int all = gasnet_nodes();

	std::cout << "hello from " << me << " of " << all << std::endl;
	
	adabs::barrier_wait();
	
	if (me == 0) {
		std::cout << "*** " << all << " processes each with " << THREADS << " worker threads each" << " ***" << std::endl;
		std::cout << "*** " << "Matrix size: " << SIZE << " ***" << std::endl;
		std::cout << "*** " << "Tile size: " << TILE_SIZE << " ***" << std::endl;
		const double matrix_memory_mb = sizeof(double) * SIZE * SIZE / 1024 / 1024;
		std::cout << "*** " << "Memory requirement per matrix: " << matrix_memory_mb << "MB" << " ***" << std::endl;
		std::cout << "*** " << "Time required to transfer 1 matrix @ 750Mbit/s: " << matrix_memory_mb * 8 / 750 << "s" << " ***" << std::endl;
	}

	adabs::barrier_wait();
	
	// result computed by a single node. used to verify results
	adabs::matrix<double, TILE_SIZE> C_seq(SIZE, SIZE);

	// compute C_seq
	std::cout << me << ": starting single node init" << std::endl;
	
	adabs::matrix<double, TILE_SIZE> A(SIZE, SIZE), B(SIZE, SIZE);
	
	init_matrixes(A, B, 0, 1);
	
	adabs::matrix<double, TILE_SIZE> A_seq(SIZE, SIZE), B_seq(SIZE, SIZE);
	
	std::cout << me << ": starting single node computation" << std::endl;

	gettimeofday(&tv1, NULL);
	compute(A, B, C_seq, A_seq, B_seq, V1, V2, 0, 1);
	gettimeofday(&tv2, NULL);
	
	std::cout << me << ": runtime (1 node): " << timediff(tv2, tv1) << std::endl;

	for (int a=0; a<SIZE/TILE_SIZE; ++a) {
		for (int b=0; b<SIZE/TILE_SIZE; ++b) {
			compare(C_seq, C_seq, a, b);
		}
	}
	
	// distributed computation
	{
		std::cout << me << ": starting distributed init" << std::endl;
		typedef adabs::distributed::matrix<double, TILE_SIZE, adabs::distributed::row_distribution<1> >  dist_matrix;
		dist_matrix *Adist = new dist_matrix(SIZE, SIZE), 
			        *A1dist = new dist_matrix(SIZE, SIZE),
			        *Bdist = new dist_matrix(SIZE, SIZE),
			        *B2dist = new dist_matrix(SIZE, SIZE),
			        *Cdist = new dist_matrix(SIZE, SIZE);
	
		Adist -> use();
		A1dist -> use();
		Bdist -> use();
		B2dist -> use();
		Cdist -> use();
		
		init_matrixes(*Adist, *Bdist, me, all);

		adabs::barrier_wait();
		std::cout << me << ": starting distributed computation" << std::endl;
		
		gettimeofday(&tv1, NULL);
		compute(*Adist, *Bdist, *Cdist, *A1dist, *B2dist, V1, V2, me, all);
		gettimeofday(&tv2, NULL);
		
		std::cout << me << ": runtime (distributed, requires reading 0.5 matrixes from remote mem.): " << timediff(tv2, tv1) << std::endl;
		
		adabs::barrier_wait();
		
		for (int a=0; a<SIZE/TILE_SIZE; ++a) {
			for (int b=0; b<SIZE/TILE_SIZE; ++b) {
				compare(C_seq, *Cdist, a, b);
			}
		}
		
		adabs::barrier_wait();
		
		Adist -> enable_reuse();
		A1dist -> enable_reuse();
		Bdist -> enable_reuse();
		B2dist -> enable_reuse();
		Cdist -> enable_reuse();
		
		Adist -> wait_for_reuse();	
		A1dist -> wait_for_reuse();
		Bdist -> wait_for_reuse();
		B2dist -> wait_for_reuse();
		Cdist -> wait_for_reuse();
		
		adabs::barrier_wait();
		
		Adist -> remove(); 
		A1dist -> remove();
		Bdist -> remove();
		B2dist -> remove();
		Cdist -> remove();
		
		adabs::barrier_wait();
	}

	//computation with remote matrix
	{
		std::cout << me << ": starting remote init" << std::endl;
		adabs::matrix<double, TILE_SIZE> A(SIZE, SIZE),
			                             A1(SIZE, SIZE),
			                             B(SIZE, SIZE),
			                             B2(SIZE, SIZE),
			                             C(SIZE, SIZE);
		
		// init my matrix
		init_matrixes(A, B, 0, 1);
			
		// all processes use the same vector
		adabs::collective::vector< adabs::pgas_addr< adabs::matrix<double, TILE_SIZE> >  > broadcast(all);
		// store reference of my local A in the vector
		broadcast.set(me, A.get_pgas_addr());

		// create remote matrix by reading the value my neighbour put in the vector
		adabs::remote_matrix<double, TILE_SIZE> Ar ( broadcast.get((me+1)%all) );
		
		std::cout << me << ": starting computation using remote A" << std::endl;
		
		adabs::barrier_wait();
		gettimeofday(&tv1, NULL);
		compute(Ar, B, C, A1, B2, V1, V2, 0, 1);
		gettimeofday(&tv2, NULL);
		adabs::barrier_wait();
		
		std::cout << me << ": runtime (remote A, requires reading 1 matrixes from remote mem.): " << timediff(tv2, tv1) << std::endl;
		
		#pragma omp parallel for
		for (int a=0; a<SIZE/TILE_SIZE; ++a) {
			for (int b=0; b<SIZE/TILE_SIZE; ++b) {
				compare(C_seq, C, a, b);
			}
		}
		adabs::barrier_wait();
	}
	
	
	return 0;
} 
