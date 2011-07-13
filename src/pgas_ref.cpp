#pragma once

#include "adabs/pgas_ref.h"

namespace adabs {

namespace pgas {

// read at node @param node on pointer @param ptr @param size bytes
void remote_read(const int node, const void *ptr, void* returnee, const int size) {
	// TODO the spec requires proper memory alignment, but I can't find
	//      any information on what is proper...
	gasnet_get((void*)returnee, node, const_cast<void*>(ptr), size); 
}

// writes data to node @param node on address @param ptr
void remote_write(const int node, void *ptr, const void* src, const int size) {
	gasnet_put(node, ptr, const_cast<void*>(src), size);
}


}

}
