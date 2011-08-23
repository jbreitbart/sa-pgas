#include "adabs/adabs.h"

//#include "adabs/matrix_base.h"
#include "adabs/pgas_addr.h"
//#include "adabs/distributed/matrix_base.h"

//#include "adabs/impl/remote_mem_management.h"

#include "adabs/collective/allocator.h"
#include "adabs/collective/pgas_addr.h"

#include <cassert>

namespace adabs {

static int _me = -1;
static int _all = -1;
static int _next = -1;
static int _prev = -1;
static bool _leader = false;

const int &me = _me;
const int &all = _all;
const int &next = _next;
const int &prev = _prev;
const bool &leader = _leader;

// variables used with busy waiting
// yes, I know, you think this is really bad, but we expect the other
// thread to react quickly (after all, we target dedicated HPC systems)
static volatile int thread_end = 0;

int get_barrier_id() {
	static int id = 0;
	id = (id+1) % 256;
	return id;
}

void barrier_wait() {
	// could use anonymous barrier here
	const int barrier_id = adabs::get_barrier_id();
	gasnet_barrier_notify(barrier_id, GASNET_BARRIERFLAG_ANONYMOUS);
	GASNET_CALL(gasnet_barrier_wait(barrier_id, GASNET_BARRIERFLAG_ANONYMOUS))
}

void exit(const int errorcode) {
	barrier_wait();
	__sync_lock_test_and_set (&thread_end, 1);
	while (thread_end != 2) {}
	barrier_wait();
	
	delete[] adabs::impl::callbacks;
	
	gasnet_exit(0);
}

static void* network(void *threadid) {
	GASNET_BEGIN_FUNCTION();
	while (thread_end != 1) {
		GASNET_CALL(gasnet_AMPoll())
	}
	__sync_lock_test_and_set (&thread_end, 2);
	pthread_exit(0);
}

void init(int *argc, char **argv[]) {
	using namespace adabs::impl;
	using namespace adabs::pgas;
	using namespace adabs::collective;
	using namespace adabs::collective::pgas;
	
	GASNET_CALL(gasnet_init (argc, argv))
	_all = gasnet_nodes();
	_me  = gasnet_mynode();
	_next = (me+1)%all;
	_prev = (me+all-1)%all;
	_leader = (me == 0);
	
	int counter = 0;

	callbacks = new gasnet_handlerentry_t[NUMBER_OF_CALLBACKS];
	/*callbacks[counter].index = MATRIX_BASE_INIT_GET;
	callbacks[counter++].fnptr = (void (*)()) &matrix_init_get;
	
	callbacks[counter].index = MATRIX_BASE_INIT_SET;
	callbacks[counter++].fnptr = (void (*)()) &matrix_init_set;

	callbacks[counter].index = MATRIX_BASE_SET;
	callbacks[counter++].fnptr = (void (*)()) &remote_set_matrix_tile;
	
	callbacks[counter].index = MATRIX_BASE_GET;
	callbacks[counter++].fnptr = (void (*)()) &remote_get_matrix_tile;

	callbacks[counter].index = DISTRIBUTED_MATRIX_BASE_DELETE;
	callbacks[counter++].fnptr = (void (*)()) &delete_matrix;
	
	callbacks[counter].index = DISTRIBUTED_MATRIX_BASE_REMOVE;
	callbacks[counter++].fnptr = (void (*)()) &remove_matrix;
	
	callbacks[counter].index = DISTRIBUTED_MATRIX_BASE_DELETE_ALL;
	callbacks[counter++].fnptr = (void (*)()) &delete_all_matrix;
	
	callbacks[counter].index = DISTRIBUTED_MATRIX_BASE_ENABLE_REUSE_ALL;
	callbacks[counter++].fnptr = (void (*)()) &reuse_all_matrix;
	
	callbacks[counter].index = DISTRIBUTED_MATRIX_BASE_REUSE;
	callbacks[counter++].fnptr = (void (*)()) &reuse_matrix;
	
	callbacks[counter].index = DISTRIBUTED_MATRIX_BASE_REUSE_REPLY;
	callbacks[counter++].fnptr = (void (*)()) &reuse_matrix_reply;
	
	callbacks[counter].index = DISTRIBUTED_MATRIX_BASE_ENABLE_REUSE_ALL_REPLY;
	callbacks[counter++].fnptr = (void (*)()) &reuse_all_matrix_reply;
	
	callbacks[counter].index = DISTRIBUTED_MATRIX_BASE_RESET_USE_FLAG;
	callbacks[counter++].fnptr = (void (*)()) &remote_reset_reuse_all_remote_counter;
	
	callbacks[counter].index = DISTRIBUTED_MATRIX_BASE_SCATTER;
	callbacks[counter++].fnptr = (void (*)()) &scatter_matrix_caller;*/
	
/*	callbacks[counter].index = MEMORY_MANAGEMENT_MALLOC;
	callbacks[counter++].fnptr = (void (*)()) &remote_malloc_real;
	
	callbacks[counter].index = MEMORY_MANAGEMENT_MALLOC_REPLY;
	callbacks[counter++].fnptr = (void (*)()) &remote_malloc_real_reply;
	
	callbacks[counter].index = MEMORY_MANAGEMENT_FREE;
	callbacks[counter++].fnptr = (void (*)()) &remote_free_real;*/
	
	callbacks[counter].index = PGAS_ADDR_SET;
	callbacks[counter++].fnptr = (void (*)()) &adabs::pgas::pgas_addr_remote_set;
	
	callbacks[counter].index = PGAS_ADDR_GET;
	callbacks[counter++].fnptr = (void (*)()) &pgas_addr_remote_get;
	
	callbacks[counter].index = COLLECTIVE_ALLOC_BROADCAST;
	callbacks[counter++].fnptr = (void (*)()) &add_to_stack;
	
	callbacks[counter].index = REMOTE_COLLECTIVE_ALLOC;
	callbacks[counter++].fnptr = (void (*)()) &remote_allocate_real;
	
	callbacks[counter].index = REMOTE_COLLECTIVE_FREE;
	callbacks[counter++].fnptr = (void (*)()) &remote_free_real;
	
	callbacks[counter].index = REMOTE_COLLECTIVE_ALLOC_REPLY;
	callbacks[counter++].fnptr = (void (*)()) &remote_malloc_real_reply;
	
	callbacks[counter].index = COLLECTIVE_PGAS_ADDR_SET;
	callbacks[counter++].fnptr = (void (*)()) &adabs::collective::pgas::pgas_addr_remote_set;
	
	assert(counter == NUMBER_OF_CALLBACKS);



	GASNET_CALL(gasnet_attach (callbacks, NUMBER_OF_CALLBACKS, gasnet_getMaxLocalSegmentSize(), 0))

	pthread_t network_thread;
	pthread_create(&network_thread, 0, network, 0);
	

}

}
