#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <pthread.h>
#include <sys/time.h>

#include "adabs/adabs.h"
#include "adabs/allocator.h"
#include "adabs/pgas_addr.h"

#include "adabs/collective/allocator.h"
#include "adabs/collective/pgas_addr.h"

#include "adabs/distributed/row_distribution.h"

#include "adabs/local.h"
#include "adabs/remote.h"
#include "adabs/collective/everywhere.h"

#include "adabs/matrix.h"

using namespace std;

const int THREADS = 4;

const int SIZE = 640;
const int BSIZE = 64;

template <typename T>
void check_matrix(const T& m) {
	for (int i=0; i<SIZE; i+=m.get_tile_size()) {
		for (int j=0; j<SIZE; j+=m.get_tile_size()) {
			//std::cout << adabs::me << ": getting " << i << ", " << j << std::endl;
			const int * ptr = m.get_tile(i, j);
			//std::cout << adabs::me << ": getting " << i << ", " << j << " - done" << std::endl;
			for (int ii=0; ii<BSIZE; ++ii) {
				for (int jj=0; jj<BSIZE; ++jj) {
					if (!(ptr[ii*BSIZE+jj] == jj+23))
						std::cout << ptr[ii*BSIZE+jj] << " should be " << jj+23 << std::endl;
					assert (ptr[ii*BSIZE+jj] == jj+23);
				}
			}
		
		}
	}
}

template <typename T>
void check_and_fill_matrix(T& m, const int start, const int stride) {
	for (int j=0; j<SIZE; j+=m.get_tile_size()) {
		for (int i=start*m.get_tile_size(); i<SIZE; i+=m.get_tile_size()*stride) {
			int * ptr = m.get_tile_unitialized(i, j);
			
			for (int ii=0; ii<BSIZE; ++ii) {
				for (int jj=0; jj<BSIZE; ++jj) {
						ptr[ii*BSIZE+jj] = jj+23;
				}
			}
			
			m.set_tile(i, j, ptr);
		}
	}

	check_matrix(m);
}

int main(int argc, char *argv[]) {
	using adabs::me;
	using adabs::all;
	using adabs::next;
	
	adabs::init(&argc, &argv);
	omp_set_num_threads(THREADS);
	
	typedef adabs::matrix<adabs::local<int> > local_matrix;
	typedef adabs::matrix<adabs::remote<int> > remote_matrix;

	local_matrix ml(SIZE, SIZE, BSIZE);
	check_and_fill_matrix (ml, 0, 1);

	adabs::matrix < adabs::collective::everywhere<int> > mev(SIZE, SIZE, BSIZE);
	check_and_fill_matrix (mev, me, all);

	adabs::vector < adabs::collective::everywhere < remote_matrix > > ma_v(all);
	remote_matrix *ma_v_ptr = ma_v.get_unitialized(me);
	ma_v_ptr[0] = ml.make_remote();
	ma_v.set(me, ma_v_ptr);
	
	check_matrix(ma_v.get(next));

	
	// test local = local assignment
	{
		adabs::matrix < adabs::local<int> > local_ma(SIZE, SIZE, BSIZE);
		check_and_fill_matrix (local_ma, 0, 1);
		
		adabs::matrix < adabs::local<int> > local_2_ma(SIZE, SIZE, BSIZE);
		
		local_2_ma = local_ma;
		
		check_matrix(local_2_ma);
		adabs::barrier_wait();
	}
	

	// test remote = local assignment
	{
		adabs::barrier_wait();
		adabs::vector < adabs::collective::everywhere < remote_matrix > > ma_v(1);
		if (me == 0) {
			local_matrix local_test(SIZE, SIZE, BSIZE);
			remote_matrix *ma_v_ptr = ma_v.get_unitialized(0);
			ma_v_ptr[0] = local_test.make_remote();
			ma_v.set(me, ma_v_ptr);
			check_matrix(local_test);
		}
		if (me == 1) {
			local_matrix local_test(SIZE, SIZE, BSIZE);
			check_and_fill_matrix (local_test, 0, 1);
			
			remote_matrix remote_test (ma_v.get(0));
			remote_test = local_test;
		}
		adabs::barrier_wait();
	}


	// test local = remote assignment
	{
		adabs::vector < adabs::collective::everywhere < remote_matrix > > ma_v(1);
		if (me == 0) {
			local_matrix local_test(SIZE, SIZE, BSIZE);
			
			remote_matrix *ma_v_ptr = ma_v.get_unitialized(0);
			ma_v_ptr[0] = local_test.make_remote();
			ma_v.set(me, ma_v_ptr);
			
			check_and_fill_matrix (local_test, 0, 1);
		}
		if (me == 1) {
			local_matrix local_test(SIZE, SIZE, BSIZE);
			
			remote_matrix remote_test (ma_v.get(0));
			local_test = remote_test;
			check_matrix(local_test);
		}
		adabs::barrier_wait();
	}


	// Gather(!)
	// test local = distributed assignment
	{
		adabs::matrix < adabs::distributed::row_distribution<int, BSIZE> > dist_ma(SIZE, SIZE, BSIZE);
		
		check_and_fill_matrix (dist_ma, me, all);
		
		check_matrix(dist_ma);
		
		if (me ==0) {
			adabs::matrix < adabs::local<int> > local_ma(SIZE, SIZE, BSIZE);
			//check_matrix(local_ma);
		}
		adabs::barrier_wait();
	}
	
	{
		adabs::pgas_addr<int> itile = adabs::allocator<int>::allocate(100*128, 128);
	
		#pragma omp parallel
		{
			if (omp_get_thread_num() == 0) {
				int * ptr = itile.get_data_unitialized();
			
				for (int i=0; i<128; ++i) {
					ptr[i] = i;
				}
			
				itile.set_data(ptr);
			}
		
			adabs::pgas_addr<int> tlocal = itile + omp_get_thread_num();
		
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
	
		adabs::allocator<int>::deallocate(itile);

		adabs::barrier_wait();
	}

	{
		adabs::collective::pgas_addr<int> ictile = adabs::collective::allocator<int>::allocate(128, 128);
	
		if (me==0) {
			int * ptr = ictile.get_data_unitialized();
	
			for (int i=0; i<128; ++i) {
				ptr[i] = i;
			}
	
			ictile.set_data(ptr);
		} else {
			const int * ptr = ictile.get_data();
			for (int i=0; i<128; ++i) {
				assert (i == ptr[i]); 
			}
		}

		adabs::barrier_wait();
		
		adabs::collective::allocator<int>::deallocate(ictile);
	}
	
	{
		adabs::distributed::row_distribution<int, BSIZE> distri(SIZE, SIZE, BSIZE, BSIZE);
	
		int inc = all*BSIZE;
		#pragma omp parallel for
		for (int i=me*BSIZE; i<SIZE; i+=inc) {
			for (int j=0; j<SIZE; j+=BSIZE) {
				int* ptr = distri.get_data_unitialized(i, j);
			
				for (int ii=0; ii<BSIZE*BSIZE; ++ii) {
					ptr[ii] = ii+23;
				}
			
				distri.set_data(i, j, ptr);
			}
		}
		
		#pragma omp parallel for
		for (int i=0; i<SIZE; i+=BSIZE) {
			for (int j=0; j<SIZE; j+=BSIZE) {
				const int* ptr = distri.get_data(i, j);
			
				for (int ii=0; ii<BSIZE*BSIZE; ++ii) {
					assert(ptr[ii] == ii+23);
				}
			}
		}
		
		adabs::barrier_wait();
	}

	std::cout << me << ": " << "Everything fine!" << std::endl;

	adabs::exit(0);
	
	return 0;
} 
