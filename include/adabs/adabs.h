#pragma once

namespace adabs {

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
