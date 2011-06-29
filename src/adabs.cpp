#include "adabs/gasnet_config.h"

#include "adabs/matrix_base.h"
#include "adabs/collective/vector_base.h"
#include "adabs/distributed/matrix_base.h"

#include "adabs/tools/tools.h"

namespace adabs {

int get_barrier_id() {
	static int id = 0;
	id = (id+1) % 256;
	return id;
}

void barrier_wait() {
	const int barrier_id = adabs::get_barrier_id();
	gasnet_barrier_notify(barrier_id, GASNET_BARRIERFLAG_ANONYMOUS);
	GASNET_CALL(gasnet_barrier_wait(barrier_id, GASNET_BARRIERFLAG_ANONYMOUS))
}

static void* network(void *threadid) {
	GASNET_BEGIN_FUNCTION();
	while (true) {
		GASNET_CALL(gasnet_AMPoll())
	}
	pthread_exit(0);
}


void init(int *argc, char **argv[]) {
	using namespace adabs::impl;
	using namespace adabs::pgas;
	using namespace adabs::collective;
	using namespace adabs::collective::pgas;
	using namespace adabs::distributed::pgas;
	
	GASNET_CALL(gasnet_init (argc, argv))
	
	const int all = gasnet_nodes();
	
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
	
	GASNET_CALL(gasnet_attach (callbacks, NUMBER_OF_CALLBACKS, gasnet_getMaxLocalSegmentSize(), 0))

	pthread_t network_thread;
	pthread_create(&network_thread, 0, network, 0);
	

}

}
