#include "adabs/adabs.h"

#include "adabs/matrix_base.h"
#include "adabs/collective/vector_base.h"
#include "adabs/distributed/matrix_base.h"

#include "adabs/impl/remote_mem_management.h"

namespace adabs {

static int _me = -1;
static int _all = -1;

const int &me = _me;
const int &all = _all;

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
	__sync_lock_test_and_set (&thread_end, 1);
	while (thread_end != 2) {}
	barrier_wait();
	
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
	using namespace adabs::impl::pgas;
	using namespace adabs::pgas;
	using namespace adabs::collective;
	using namespace adabs::collective::pgas;
	using namespace adabs::distributed::pgas;
	
	GASNET_CALL(gasnet_init (argc, argv))
	_all = gasnet_nodes();
	_me  = gasnet_mynode();
	
	vector_base::global_com = new vector_base*[all];
	for (int i=0; i<all; ++i) vector_base::global_com[i] = 0; 

	callbacks = new gasnet_handlerentry_t[NUMBER_OF_CALLBACKS];
	callbacks[0].index = MATRIX_BASE_INIT_GET;
	callbacks[0].fnptr = (void (*)()) &matrix_init_get;
	
	callbacks[1].index = MATRIX_BASE_INIT_SET;
	callbacks[1].fnptr = (void (*)()) &matrix_init_set;

	callbacks[2].index = COLLECTIVE_VECTOR_GLOBAL_COM_SET;
	callbacks[2].fnptr = (void (*)()) &set_global_com;

	callbacks[3].index = COLLECTIVE_VECTOR_SET;
	callbacks[3].fnptr = (void (*)()) &remote_set_vector_element;

	callbacks[4].index = MATRIX_BASE_SET;
	callbacks[4].fnptr = (void (*)()) &remote_set_matrix_tile;
	
	callbacks[5].index = MATRIX_BASE_GET;
	callbacks[5].fnptr = (void (*)()) &remote_get_matrix_tile;

	callbacks[6].index = DISTRIBUTED_MATRIX_BASE_DELETE;
	callbacks[6].fnptr = (void (*)()) &delete_matrix;
	
	callbacks[7].index = DISTRIBUTED_MATRIX_BASE_REMOVE;
	callbacks[7].fnptr = (void (*)()) &remove_matrix;
	
	callbacks[8].index = DISTRIBUTED_MATRIX_BASE_DELETE_ALL;
	callbacks[8].fnptr = (void (*)()) &delete_all_matrix;
	
	callbacks[9].index = DISTRIBUTED_MATRIX_BASE_ENABLE_REUSE_ALL;
	callbacks[9].fnptr = (void (*)()) &reuse_all_matrix;
	
	callbacks[10].index = DISTRIBUTED_MATRIX_BASE_REUSE;
	callbacks[10].fnptr = (void (*)()) &reuse_matrix;
	
	callbacks[11].index = DISTRIBUTED_MATRIX_BASE_REUSE_REPLY;
	callbacks[11].fnptr = (void (*)()) &reuse_matrix_reply;
	
	callbacks[12].index = DISTRIBUTED_MATRIX_BASE_ENABLE_REUSE_ALL_REPLY;
	callbacks[12].fnptr = (void (*)()) &reuse_all_matrix_reply;
	
	callbacks[13].index = DISTRIBUTED_MATRIX_BASE_RESET_USE_FLAG;
	callbacks[13].fnptr = (void (*)()) &remote_reset_reuse_all_remote_counter;
	
	callbacks[14].index = DISTRIBUTED_MATRIX_BASE_SCATTER;
	callbacks[14].fnptr = (void (*)()) &scatter_matrix_caller;
	
	callbacks[15].index = MEMORY_MANAGEMENT_MALLOC;
	callbacks[15].fnptr = (void (*)()) &remote_malloc_real;
	
	callbacks[16].index = MEMORY_MANAGEMENT_MALLOC_REPLY;
	callbacks[16].fnptr = (void (*)()) &remote_malloc_real_reply;
	
	callbacks[17].index = MEMORY_MANAGEMENT_FREE;
	callbacks[17].fnptr = (void (*)()) &remote_free_real;
	
	GASNET_CALL(gasnet_attach (callbacks, NUMBER_OF_CALLBACKS, gasnet_getMaxLocalSegmentSize(), 0))

	pthread_t network_thread;
	pthread_create(&network_thread, 0, network, 0);
	

}

}
