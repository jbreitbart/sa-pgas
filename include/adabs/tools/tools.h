#pragma once

#include <stdlib.h>
#include <iostream>
#include <sys/time.h>

inline double timediff(timeval tv2, timeval tv1) {
	return (double) (tv2.tv_sec - tv1.tv_sec) + ((double) (tv2.tv_usec - tv1.tv_usec) / 1000000.0);
}

namespace adabs {

namespace tools {

#define GASNET_CALL(a) if (int error = a != GASNET_OK) adabs::tools::kill(error);

inline void kill(int error) {
	std::cerr << "GASNet error " << error << " on Node " << gasnet_mynode() << std::endl; 
}


/**
 * Just a posix memalign wrapper returning memory aligned by 16.
 * Memory should be deleted with the standard free.
 */
template <typename T>
void memalign_wrapper(T* &ptr, const int size) {
	if (posix_memalign((void**) &ptr, 16, size*sizeof(T)) != 0) {
		std::cerr << "error" << std::endl;
	}
}

}

}
