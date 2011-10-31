#include "adabs/impl/pgas_addr.h"
#include "adabs/pgas_addr.h"


namespace adabs {

namespace impl {

std::list<remote_get_thread_arg> global_requests;
static std::list<remote_get_thread_arg> local_requests;

static inline bool process_single_request(remote_get_thread_arg arg) {
	using namespace adabs::tools;
	
	// wait until flag is set
	volatile int* reader = (volatile int*) ((char*)arg.local + arg.batch_mem_size);

	//#pragma omp critical
	//std::cout << "remote waiting on pointer " << (void*)reader << std::endl;
	bool returnee = (*reader == adabs::pgas_addr<void>::FULL);
	
	if (returnee == false) return false;
	
	//while (*reader != adabs::pgas_addr<void>::FULL) {}
	//#pragma omp critical
	//std::cout << "remote waiting on pointer " << (void*)reader << " done " << std::endl;
	
	// sent data back
	void* buf = (char*)arg.local;

	int* data = (int*)arg.remote;
	int* flag = (int*)((char*)arg.remote + arg.batch_mem_size);

	GASNET_CALL(
	            gasnet_AMRequestLong2(arg.dest, adabs::impl::PGAS_ADDR_SET,
	                                  buf, arg.batch_mem_size, data,
	                                  get_low(flag),
	                                  get_high(flag)
	                                 )
	           )
	
	return true;
}


void process_requests() {
	//copy global to local list (lock)
	#pragma omp critical (global_requests)
	{
		local_requests.splice (local_requests.begin(), global_requests,
		                       global_requests.begin(),
		                       global_requests.end()
		                      );
	}
	
	//for all elements in local list
	for ( std::list<remote_get_thread_arg>::iterator it = local_requests.begin();
	      it != local_requests.end();
	    ) {
		//	process element / delete if true
		bool done = process_single_request(*it);
		
		std::list<remote_get_thread_arg>::iterator prev = it;
		
		++it;
		
		if (done) local_requests.erase(prev);
	}
}

void end_requests() {
	assert (local_requests.empty());
	assert (global_requests.empty());
}

}

}
