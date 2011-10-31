#pragma once

#include <list>
#include <cassert>

#include "adabs/adabs.h"

namespace adabs {

namespace impl {

/**
 * Pthread argument class
 */
struct remote_get_thread_arg {
	void *local;
	const int batch_mem_size;
	void *remote;
	const gasnet_node_t dest;
	const int flag_diff;
	
	remote_get_thread_arg(void *_local, 
	                      const int _batch_mem_size,
	                      void *_remote,
	                      gasnet_node_t _dest,
	                      const int _flag_diff
	                      ) : local(_local),
	                          batch_mem_size(_batch_mem_size),
	                          remote(_remote),
	                          dest(_dest),
	                          flag_diff(_flag_diff) {}
};

extern std::list<remote_get_thread_arg> global_requests;

void process_requests(); 
void end_requests();


}

}
