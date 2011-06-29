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
const int SIZE            = 4096*8;
const int THREADS         = 4;
const int ITERATIONS      = 100;


const int LEFT = 1;
const int RIGHT = 2;


template<int TYPE, int START, int UNUSED>
static inline void compute_tile_top(      double * restrict cur,
                                    const double * restrict cur_old,
                                    const double * restrict left,
                                    const double * restrict right,
                                    const double * restrict _top) {
	const double * restrict top;
	if (START==0)
		top = &_top[(TILE_SIZE-1)*TILE_SIZE];
	else
		top = &cur[(START-1)*TILE_SIZE];
	
	if (TYPE!=LEFT)
		cur[START*TILE_SIZE] = (left[START*TILE_SIZE+TILE_SIZE-1]
				 +cur_old[START*TILE_SIZE+1]
				 +top[0]
				 +cur_old[START*TILE_SIZE+TILE_SIZE])
				 *0.25;
		     
	for (int j=1; j<TILE_SIZE-1; ++j) {
		cur[START*TILE_SIZE+j] = (cur[START*TILE_SIZE+j-1]
			     +cur_old[START*TILE_SIZE+j+1]
			     +top[j]
			     +cur_old[START*TILE_SIZE+TILE_SIZE+j])
			     *0.25;
	}
	
	if (TYPE!=RIGHT)
		cur[START*TILE_SIZE+TILE_SIZE-1] = (START*TILE_SIZE+cur[TILE_SIZE-1-1]
				           +right[START*TILE_SIZE]
				           +top[TILE_SIZE-1]
				           +cur_old[TILE_SIZE+TILE_SIZE-1])
				           *0.25;
}

template<int TYPE, int UNUSED, int END>
static inline void compute_tile_mid(      double * restrict cur,
                                    const double * restrict cur_old,
                                    const double * restrict left,
                                    const double * restrict right) {
	for (int i=1; i<END-1; ++i) {
		
		if (TYPE!=LEFT)
			cur[i*TILE_SIZE] = (left[i*TILE_SIZE+TILE_SIZE-1]
				                 +cur_old[i*TILE_SIZE+1]
				                 +cur[(i-1)*TILE_SIZE]
				                 +cur_old[(i+1)*TILE_SIZE])
				                 *0.25;
		  
		for (int j=1; j<TILE_SIZE-1; ++j) {
			cur[i*TILE_SIZE+j] = (cur[i*TILE_SIZE+j-1]
			                     +cur_old[i*TILE_SIZE+j+1]
			                     +cur[(i-1)*TILE_SIZE+j]
			                     +cur_old[(i+1)*TILE_SIZE+j])
			                     *0.25;
		}
		
		if (TYPE!=RIGHT)
			cur[i*TILE_SIZE+TILE_SIZE-1] = (cur[i*TILE_SIZE+TILE_SIZE-1-1]
				                           +right[i*TILE_SIZE]
				                           +cur[(i-1)*TILE_SIZE+TILE_SIZE-1]
				                           +cur_old[(i+1)*TILE_SIZE+TILE_SIZE-1])
				                           *0.25;
	}
}

template<int TYPE, int UNUSED, int END>
static inline void compute_tile_bot(      double * restrict cur,
                                    const double * restrict cur_old,
                                    const double * restrict left,
                                    const double * restrict right,
                                    const double * restrict _bot) {
	const double * restrict bot;
	if (END==TILE_SIZE)
		bot = &_bot[0];
	else
		bot = &cur_old[END];
		
	if (TYPE!=LEFT)
		cur[(END-1)*TILE_SIZE] = (left[(END-1)*TILE_SIZE+TILE_SIZE-1]
			                           +cur_old[(END-1)*TILE_SIZE+1]
			                           +cur[(END-1-1)*TILE_SIZE]
			                           +bot[0])
			                           *0.25;
	  
	for (int j=1; j<TILE_SIZE-1; ++j) {
		cur[(END-1)*TILE_SIZE+j] = (cur[(END-1)*TILE_SIZE+j-1]
		                     +cur_old[(END-1)*TILE_SIZE+j+1]
		                     +cur[(END-1-1)*TILE_SIZE+j]
		                     +bot[j])
		                     *0.25;
	}
	
	if (TYPE!=RIGHT)
		cur[(END-1)*TILE_SIZE+TILE_SIZE-1] = (cur[(END-1)*TILE_SIZE+TILE_SIZE-1-1]
			                           +right[(END-1)*TILE_SIZE]
			                           +cur[(END-1-1)*TILE_SIZE+TILE_SIZE-1]
			                           +bot[TILE_SIZE-1])
			                           *0.25;
}

template<typename T>
static inline void compute_matrix(T &old_ma, T &cur_ma, int me, int all) {
	int b = SIZE/TILE_SIZE/all * me + omp_get_thread_num();
	const int end = (me+1==all) ? SIZE/TILE_SIZE-1 : SIZE/TILE_SIZE/all * (me+1);
	const int jump = THREADS;
	
	//int b=me*THREADS + omp_get_thread_num();
	//const int end = SIZE/TILE_SIZE-1;
	//const int jump = 	all*THREADS;
	
	//std::cout << "compute info. start: " << b << "; end: " << end << "; jump: " << jump << std::endl;
	if (b==0) {
		// sonderbehandlung obere zeile
		//std::cout << me << ": b = " << b << std::endl;
		{
				  double * restrict cur = cur_ma.get_tile_unitialized(0, 0);

			const double * restrict right   = old_ma.get_tile(1,0); // old
			const double * restrict bot     = old_ma.get_tile(0,1); // old
			const double * restrict cur_old = old_ma.get_tile(0,0); // old
		
			compute_tile_top<LEFT, 1, TILE_SIZE>(cur, cur_old, 0, right, 0);
			compute_tile_mid<LEFT, 1, TILE_SIZE>(cur, cur_old, 0, right);
			compute_tile_bot<LEFT, 1, TILE_SIZE>(cur, cur_old, 0, right, bot);
		
			cur_ma.set_tile(cur, 0, 0);
		}
		
		for (int a=1; a<SIZE/TILE_SIZE-1; ++a) {
			//std::cout << me << ": b = " << b << ", a = " << a << std::endl;
			
			      double * restrict cur     = cur_ma.get_tile_unitialized(a,0);
			const double * restrict left    = cur_ma.get_tile(a-1,0); // new
			const double * restrict right   = old_ma.get_tile(a+1,0); // old
			const double * restrict bot     = old_ma.get_tile(a,1); // old
			const double * restrict cur_old = old_ma.get_tile(a,0); // old
			
			compute_tile_top<0, 1, TILE_SIZE>(cur, cur_old, left, right, 0);
			compute_tile_mid<0, 1, TILE_SIZE>(cur, cur_old, left, right);
			compute_tile_bot<0, 1, TILE_SIZE>(cur, cur_old, left, right, bot);
			
			cur_ma.set_tile(cur, a, 0);
		}
		
		{
				  double * restrict cur     = cur_ma.get_tile_unitialized(SIZE/TILE_SIZE-1 ,0);
			const double * restrict left    = cur_ma.get_tile(SIZE/TILE_SIZE-2, 0); // new
			const double * restrict bot     = old_ma.get_tile(SIZE/TILE_SIZE-1, 1); // old
			const double * restrict cur_old = old_ma.get_tile(SIZE/TILE_SIZE,0); // old
		
			compute_tile_top<RIGHT, 1, TILE_SIZE>(cur, cur_old, left, 0, 0);
			compute_tile_mid<RIGHT, 1, TILE_SIZE>(cur, cur_old, left, 0);
			compute_tile_bot<RIGHT, 1, TILE_SIZE>(cur, cur_old, left, 0, bot);
			
			cur_ma.set_tile(cur, SIZE/TILE_SIZE-1, 0);
		}
		
		b+=jump;
	}
	
	for (;b<end; b+=jump) {
		//std::cout << me << ":l b = " << b << std::endl;
		{
				  double * restrict cur = cur_ma.get_tile_unitialized(0,b);
			const double * restrict right   = old_ma.get_tile(1,b); // old
			const double * restrict top     = old_ma.get_tile(0,b-1); // old
			const double * restrict bot     = old_ma.get_tile(0,b+1); // old
			const double * restrict cur_old = old_ma.get_tile(0,b); // old
		
			compute_tile_top<LEFT, 0, TILE_SIZE>(cur, cur_old, 0, right, top);
			compute_tile_mid<LEFT, 0, TILE_SIZE>(cur, cur_old, 0, right);
			compute_tile_bot<LEFT, 0, TILE_SIZE>(cur, cur_old, 0, right, bot);
			
			cur_ma.set_tile(cur, 0, b);
		}
		
		for (int a=1; a<SIZE/TILE_SIZE-1; ++a) {
			
			      double * restrict cur = cur_ma.get_tile_unitialized(a, b);
			const double * restrict left    = cur_ma.get_tile(a-1, b); // new
			const double * restrict right   = old_ma.get_tile(a+1, b); // old
			const double * restrict top  = old_ma.get_tile(a,b-1); // old
			const double * restrict bot  = old_ma.get_tile(a, b+1); // old
			const double * restrict cur_old = old_ma.get_tile(a, b); // old
			
			compute_tile_top<0, 0, TILE_SIZE>(cur, cur_old, left, right, top);
			compute_tile_mid<0, 0, TILE_SIZE>(cur, cur_old, left, right);
			compute_tile_bot<0, 0, TILE_SIZE>(cur, cur_old, left, right, bot);
			
			//std::cout << me << ":l b = " << b << ", a = " << a << std::endl;
			cur_ma.set_tile(cur, a, b);
		}
		
		{
				  double * restrict cur = cur_ma.get_tile_unitialized(SIZE/TILE_SIZE-1 ,b);
			//std::cout << me << ":ll b = " << b << ", a = " << SIZE/TILE_SIZE-1 << " 1" << std::endl;
			const double * restrict left    = cur_ma.get_tile(SIZE/TILE_SIZE-2, b); // new
			//std::cout << me << ":ll b = " << b << ", a = " << SIZE/TILE_SIZE-1 << " 2" << std::endl;
			const double * restrict top  = old_ma.get_tile(SIZE/TILE_SIZE-1, b-1); // old
			//std::cout << me << ":ll b = " << b << ", a = " << SIZE/TILE_SIZE-1 << " 3" << std::endl;
			const double * restrict bot  = old_ma.get_tile(SIZE/TILE_SIZE-1, b+1); // old
			//std::cout << me << ":ll b = " << b << ", a = " << SIZE/TILE_SIZE-1 << " 4" << std::endl;
			const double * restrict cur_old = old_ma.get_tile(SIZE/TILE_SIZE-1, b); // old
			//std::cout << me << ":ll b = " << b << ", a = " << SIZE/TILE_SIZE-1 << " 5" << std::endl;
			
			compute_tile_top<RIGHT, 0, TILE_SIZE>(cur, cur_old, left, 0, top);
			compute_tile_mid<RIGHT, 0, TILE_SIZE>(cur, cur_old, left, 0);
			compute_tile_bot<RIGHT, 0, TILE_SIZE>(cur, cur_old, left, 0, bot);
			
			//std::cout << me << ":ll b = " << b << ", a = " << SIZE/TILE_SIZE-1 << std::endl;
			cur_ma.set_tile(cur, SIZE/TILE_SIZE-1, b);
			//std::cout << me << ":ll b = " << b << ", a = " << SIZE/TILE_SIZE-1 << " - done" << std::endl;
		}
	}

	if (b==SIZE/TILE_SIZE-1){
		// sonderbehandlung letzte zeile
		{
				  double * restrict cur = cur_ma.get_tile_unitialized(0, SIZE/TILE_SIZE-1);
					  
			const double * restrict right   = old_ma.get_tile(1,SIZE/TILE_SIZE-1); // old
			const double * restrict top  = old_ma.get_tile(0,SIZE/TILE_SIZE-1-1); // old
			const double * restrict cur_old = old_ma.get_tile(0,SIZE/TILE_SIZE-1); // old
			
			compute_tile_top<LEFT, 0, TILE_SIZE-1>(cur, cur_old, 0, right, top);
			compute_tile_mid<LEFT, 0, TILE_SIZE-1>(cur, cur_old, 0, right);
			compute_tile_bot<LEFT, 0, TILE_SIZE-1>(cur, cur_old, 0, right, 0);
			
			cur_ma.set_tile(cur, 0, SIZE/TILE_SIZE-1);
		}
		
		for (int a=1; a<SIZE/TILE_SIZE-1; ++a) {
			      double * restrict cur = cur_ma.get_tile_unitialized(a, SIZE/TILE_SIZE-1);
			const double * restrict left    = cur_ma.get_tile(a-1, SIZE/TILE_SIZE-1); // new
			const double * restrict top  = old_ma.get_tile(a, SIZE/TILE_SIZE-1-1); // old
			const double * restrict right   = old_ma.get_tile(a+1, SIZE/TILE_SIZE-1); // old
			const double * restrict cur_old = old_ma.get_tile(a, SIZE/TILE_SIZE-1); // old
			
			compute_tile_top<0, 0, TILE_SIZE-1>(cur, cur_old, left, right, top);
			compute_tile_mid<0, 0, TILE_SIZE-1>(cur, cur_old, left, right);
			compute_tile_bot<0, 0, TILE_SIZE-1>(cur, cur_old, left, right, 0);
			
			cur_ma.set_tile(cur, a, SIZE/TILE_SIZE-1);
		}
		
		{
				  double * restrict cur = cur_ma.get_tile_unitialized(SIZE/TILE_SIZE-1 ,SIZE/TILE_SIZE-1);
			const double * restrict left = cur_ma.get_tile(SIZE/TILE_SIZE-2, SIZE/TILE_SIZE-1); // new
			const double * restrict top = old_ma.get_tile(SIZE/TILE_SIZE-1, SIZE/TILE_SIZE-1-1); // old
			const double * restrict cur_old = old_ma.get_tile(SIZE/TILE_SIZE-1, SIZE/TILE_SIZE-1); // old
			
			compute_tile_top<RIGHT, 0, TILE_SIZE-1>(cur, cur_old, left, 0, top);
			compute_tile_mid<RIGHT, 0, TILE_SIZE-1>(cur, cur_old, left, 0);
			compute_tile_bot<RIGHT, 0, TILE_SIZE-1>(cur, cur_old, left, 0, 0);
			
			cur_ma.set_tile(cur, SIZE/TILE_SIZE-1, SIZE/TILE_SIZE-1);
		}
	}
} 

template <typename T>
static void fill_matrix(T& ma) {
	for (int b=0; b<SIZE/TILE_SIZE; ++b) {
		for (int a=0; a<SIZE/TILE_SIZE; ++a) {
			if (ma.is_local(a, b)) {
				double * restrict Aptr = ma.get_tile_unitialized(a,b);
				for (int i=0; i<TILE_SIZE; ++i) {
					#pragma vector aligned
					for (int j=0; j<TILE_SIZE; ++j) {
						Aptr[i*TILE_SIZE+j] = i;
					}
				}

				ma.set_tile(Aptr, a, b);
			}
		}
	}
}

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

int main(int argc, char *argv[]) {
	timeval tv1, tv2;

	adabs::init(&argc, &argv);
	gasnet_set_waitmode(GASNET_WAIT_BLOCK);
	
	omp_set_num_threads(THREADS);
	
	const int me = gasnet_mynode();
	const int all = gasnet_nodes();

	std::cout << "hello from " << me << " of " << all << std::endl;
	
	adabs::barrier_wait();
	
	typedef adabs::distributed::matrix<double, TILE_SIZE, adabs::distributed::row_distribution<SIZE/TILE_SIZE/2> > matrixT;
	
	std::vector< matrixT* > mas(all+2);
	for (int i=0; i<all+2; ++i) {
		mas[i] = new matrixT(SIZE, SIZE);
		mas[i] -> use();
		adabs::barrier_wait();
	}
	
	std::cout << me << ": done creating" << std::endl;
	adabs::barrier_wait();
	
	// init all mas with data
	//if (me==0) {
		for (int maI=0; maI<all+2; ++maI) {
			std::cout << me << ": " << maI << std::endl;
			fill_matrix(*mas[maI]);
		}
		std::cout << me << ": matrix filled" << std::endl;
	//}
	
	adabs::barrier_wait();
	for (int maI=1; maI<all+2; ++maI) {
		mas[maI] -> enable_reuse(); 
	}
	
	for (int maI=1; maI<all+2; ++maI) {
		std::cout << me << ": waiting for reuse " << maI << std::endl;
		mas[maI] -> wait_for_reuse(); 
	}
	
	
	adabs::barrier_wait();
	gettimeofday(&tv1, NULL);
	for (int i=0; i<ITERATIONS; ++i) {
		matrixT &old = *mas[i%(all+2)];
		matrixT &cur = *mas[(i+1)%(all+2)];
		
		cur.wait_for_reuse();
		cur.use();
		
		//std::cout << me << ": START " << i << " =================" << std::endl;
		
		#pragma omp parallel
		compute_matrix(old, cur, me, all);
		
		//std::cout << me << ": END " << i << " =================" << std::endl;
		
		old.enable_reuse();
		
	}
	gettimeofday(&tv2, NULL);
	std::cout << me << ": runtime (dist): " << timediff(tv2, tv1) << std::endl;
	
	adabs::barrier_wait();
	
	const int cur_index = (ITERATIONS)%(all+2);
	
	for (int i=0; i<all+2; ++i) {
		if (i!=cur_index)
			delete mas[i];
	}
	
	std::cout << me << ": starting sequential computation" << std::endl;
	
	adabs::matrix<double, TILE_SIZE> ma0_seq(SIZE, SIZE), ma1_seq(SIZE, SIZE);
	
	fill_matrix(ma0_seq);
	fill_matrix(ma1_seq);

	adabs::matrix<double, TILE_SIZE> *oldptr = &ma0_seq;
	adabs::matrix<double, TILE_SIZE> *curptr = &ma1_seq;
	
	gettimeofday(&tv1, NULL);
	for (int i=0; i<ITERATIONS; ++i) {
		adabs::matrix<double, TILE_SIZE> &old = *oldptr;
		adabs::matrix<double, TILE_SIZE> &cur = *curptr;
		
		cur.reuse();
		
		//std::cout << me << ": START " << i << " =================" << std::endl;
		
		#pragma omp parallel
		compute_matrix(old, cur, 0, 1);
		
		//std::cout << me << ": END " << i << " =================" << std::endl;
		
		adabs::matrix<double, TILE_SIZE> *temp = oldptr;
		oldptr = curptr;
		curptr = temp;
	}
	gettimeofday(&tv2, NULL);
	std::cout << me << ": runtime (sequ): " << timediff(tv2, tv1) << std::endl;
	
	adabs::barrier_wait();
	
	for (int a=0; a<SIZE/TILE_SIZE; ++a) {
		for (int b=0; b<SIZE/TILE_SIZE; ++b) {
			//compare(*mas[cur_index], *oldptr, a, b);
		}
	}

	adabs::barrier_wait();


	return 0;
} 

