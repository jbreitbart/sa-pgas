#pragma once

#include "adabs/gasnet_config.h"
#include "tools/tools.h"

#include "adabs/impl/remote_mem_management.h"
#include "adabs/pgas_addr.h"


namespace adabs {

/**
 * The number of the current node. Valid after adabs::init() was called.
 */
extern const int &me;

/**
 * The amount of nodes available. Valid after adabs::init() was called.
 */
extern const int &all;

/**
 * A node next in line, for which communication may be faster than other
 * arbitary nodes.
 */
extern const int &next;

/**
 * A node previous to you, for which communication may be faster than
 * other arbitary nodes.
 */
extern const int &prev;

/**
 * Returns a barrier ID that can be used in the next GASNet barrier
 * call.
 */
int get_barrier_id();

/**
 * Inits the sa-pgas system. 
 */
void init(int *argc, char **argv[]);

/**
 * End the program and returns the exitcode. This must be called on all
 * nodes and includes a barrier synchronization (required by GASNet
 * spec)..
 */
void exit(int exitcode);

/**
 * Creates and waits at the barrier until every node has reached the
 * barrier.
 */
void barrier_wait();

/**
 * Allocates @param size bytes at node @param node and returns a
 * valid pgas_addr.
 */
template<typename T>
pgas_addr<T> remote_malloc(const int node, const size_t size) {
	return pgas_addr<T>(node,
	                    static_cast<T*>(impl::remote_malloc(node, size*sizeof(T)))
	                   );
}

/**
 * Frees memory on a remote node.
 */
template<typename T>
void remote_free(const pgas_addr<T> &addr) {
	impl::remote_free(addr.get_node(), addr.get_ptr());
}

}
