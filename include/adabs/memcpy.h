#pragma once

#include <cassert>
#include <cstring>

#include "adabs/adabs.h"
#include "adabs/pgas_addr.h"

namespace adabs {

namespace pgas {
inline void pgas_memcpy (gasnet_token_t token, void *buf, size_t nbytes);
}


template <typename T>
void memcpy (pgas_addr<T> destination, pgas_addr<T> source, const int num) {
	const int NETWORK_BATCHES=1024*4;
	
	assert (destination.is_local() || source.is_local());
	assert (destination.get_batch_size() == source.get_batch_size());
	
	char *source_endptr = (char*)(source + (num)).get_raw_pointer();
	char *sourceptr = (char*)source.get_raw_pointer();
	char *destptr = (char*)destination.get_raw_pointer();
	
	if (destination.is_local() && source.is_local()) {
		// both local, just call memcpy
		const int nb_bytes = source_endptr - sourceptr;
		std::memcpy (destptr, sourceptr, nb_bytes);
		return;
	}
	
	if (destination.is_local()) {
		int i=0;
		while (sourceptr < source_endptr) {
			int nb_bytes = source_endptr - sourceptr;
			if (nb_bytes > NETWORK_BATCHES) nb_bytes = NETWORK_BATCHES;
			
			//std::cout << source.get_node() << "," << (void*)sourceptr << " to " << (void*)destptr << ", " << nb_bytes << ", " << i << std::endl;
			gasnet_get_nbi_bulk (destptr, source.get_node(), sourceptr, nb_bytes);
			
			sourceptr += NETWORK_BATCHES;
			destptr += NETWORK_BATCHES;
			++i;
		}
		
		gasnet_wait_syncnbi_all();
		__sync_synchronize();
		return;
	}
	
	if (source.is_local()) {
		while (sourceptr < source_endptr) {
		
			int nb_bytes = source_endptr - sourceptr;
			if (nb_bytes > NETWORK_BATCHES) nb_bytes = NETWORK_BATCHES;
			
			GASNET_CALL(gasnet_AMRequestLong0(destination.get_node(),
											   adabs::impl::MEMCPY,
											   sourceptr,
											   nb_bytes,
											   destptr
											  )
					  )
			sourceptr += NETWORK_BATCHES;
			destptr += NETWORK_BATCHES;
		}
		
		return;
	}
	
}

namespace pgas {
inline void pgas_memcpy (gasnet_token_t token, void *buf, size_t nbytes) {
	//std::cout << "wrote data from " << buf << " to " << (int*)((char*)buf + nbytes) << " value " << *(int*)buf << " - " << nbytes << std::endl;
	__sync_synchronize();
}	
}

}
