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

template <typename T>
void check_matrix(const T& m) {

	for (int i=0; i<640; i+=m.get_tile_size()) {
		for (int j=0; j<640; j+=m.get_tile_size()) {
			const int * ptr = m.get_tile(i, j);
			for (int ii=0; ii<64; ++ii) {
				for (int jj=0; jj<64; ++jj) {
					if (!(ptr[ii*64+jj] == jj+23))
						std::cout << ptr[ii*64+jj] << " should be " << jj+23 << std::endl;
					assert (ptr[ii*64+jj] == jj+23);
				}
			}
			
		}
	}
}

template <typename T>
void check_and_fill_matrix(T& m, const int start, const int stride) {
	for (int i=start*m.get_tile_size(); i<640; i+=m.get_tile_size()*stride) {
		for (int j=0; j<640; j+=m.get_tile_size()) {
			int * ptr = m.get_tile_unitialized(i, j);
			for (int ii=0; ii<64; ++ii) {
				for (int jj=0; jj<64; ++jj) {
					ptr[ii*64+jj] = jj+23;
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
	
	local_matrix ml(640, 640, 64);
	check_and_fill_matrix (ml, 0, 1);
	
	adabs::matrix < adabs::collective::everywhere<int> > mev(640, 640, 64);
	check_and_fill_matrix (mev, me, all);

	adabs::vector < adabs::collective::everywhere < remote_matrix > > ma_v(all);
	remote_matrix *ma_v_ptr = ma_v.get_unitialized(me);
	ma_v_ptr[0] = ml.make_remote();
	ma_v.set(me, ma_v_ptr);
	
	check_matrix(ma_v.get(next));
	
	// test local = local assignment
	{
		adabs::matrix < adabs::local<int> > local_ma(640, 640, 64);
		check_and_fill_matrix (local_ma, 0, 1);
		
		adabs::matrix < adabs::local<int> > local_2_ma(640, 640, 64);
		
		local_2_ma = local_ma;
		
		check_matrix(local_2_ma);
	}
	
	// test remote = local assignment
	{
		adabs::barrier_wait();
		adabs::vector < adabs::collective::everywhere < remote_matrix > > ma_v(1);
		if (me == 0) {
			local_matrix local_test(640, 640, 64);
			remote_matrix *ma_v_ptr = ma_v.get_unitialized(0);
			ma_v_ptr[0] = local_test.make_remote();
			ma_v.set(me, ma_v_ptr);
			check_matrix(local_test);
		}
		if (me == 1) {
			local_matrix local_test(640, 640, 64);
			check_and_fill_matrix (local_test, 0, 1);
			
			remote_matrix remote_test (ma_v.get(0));
			remote_test = local_test;
		}
	}
#if 0	
	// test local = remote assignment
	{
		adabs::barrier_wait();
		adabs::vector < adabs::collective::everywhere < remote_matrix > > ma_v(1);
		if (me == 0) {
			local_matrix local_test(640, 640, 64);
			remote_matrix *ma_v_ptr = ma_v.get_unitialized(0);
			ma_v_ptr[0] = local_test.make_remote();
			ma_v.set(me, ma_v_ptr);
			
			check_and_fill_matrix (local_test, 0, 1);
		}
		if (me == 1) {
			local_matrix local_test(640, 640, 64);
			
			remote_matrix remote_test (ma_v.get(0));
			local_test = remote_test;
			check_matrix(local_test);
		}
	}
#endif	
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

	adabs::collective::allocator<int>::deallocate(ictile);
	
	adabs::distributed::row_distribution<int, 64> distri(128*2, 128*2, 64);
	
	const int inc = all*64;
	#pragma omp parallel for
	for (int i=me*64; i<128*2; i+=inc) {
		for (int j=0; j<128*2; j+=64) {
			int* ptr = distri.get_data_unitialized(i, j);
			
			for (int ii=0; ii<64; ++ii) {
				ptr[ii] = ii+23;
			}
			
			distri.set_data(i, j, ptr);
		}
	}

	const int start = (me+1)%all;
	#pragma omp parallel for
	for (int i=start*64; i<128; i+=inc) {
		for (int j=0; j<128; j+=all*64) {
			const int* ptr = distri.get_data(i, j);
			
			for (int ii=0; ii<64; ++ii) {
				assert(ptr[ii] == ii+23);
			}
		}
	}
	
	
	std::cout << me << ": " << "Everything fine!" << std::endl;

	adabs::exit(0);
	
	return 0;
} 
