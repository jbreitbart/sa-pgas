#pragma once

namespace adabs {

int get_barrier_id();

void init(int *argc, char **argv[]);

void barrier_wait();

}
