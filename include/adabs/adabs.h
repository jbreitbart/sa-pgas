#pragma once

#include "adabs/gasnet_config.h"
#include "tools/tools.h"

//#include "adabs/impl/remote_mem_management.h"


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
 * Names the leader of the current group
 */
extern const bool &leader;

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

}
