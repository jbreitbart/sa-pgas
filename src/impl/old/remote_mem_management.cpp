#if 0
#include "adabs/impl/remote_mem_management.h"

#include <cstddef>

#include "adabs/adabs.h"

#include "adabs/impl/callbacks.h"
#include "adabs/tools/ptr_divider.h"


namespace adabs {
namespace impl {

void* remote_malloc(const int node, const std::size_t size) {
	using namespace adabs::tools;
	
	volatile long returnee;
	returnee = -1;
	
	 // start remote thread and allocate memory
	GASNET_CALL(gasnet_AMRequestShort3(node, adabs::impl::MEMORY_MANAGEMENT_MALLOC,
	                                   get_low(&returnee),
	                                   get_high(&returnee),
	                                   size
	                                   )
	           )
	 
	//wait until returnee != -1
	while (returnee == -1) {}

	return (void*)(returnee);
}

void remote_free(const int node, void* ptr) {
	using namespace adabs::tools;
	
	if (ptr == 0) return;
	
	GASNET_CALL(gasnet_AMRequestShort2(node, adabs::impl::MEMORY_MANAGEMENT_FREE,
	                                   get_low(ptr),
	                                   get_high(ptr)
	                                   )
	           )
}


namespace pgas {

void remote_malloc_real(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                              gasnet_handlerarg_t arg1,
                                              gasnet_handlerarg_t arg2
                       ) {
	using namespace adabs::tools;
	
	void* returnee = malloc ((size_t)arg2);
	
	GASNET_CALL(gasnet_AMReplyShort4(token, adabs::impl::MEMORY_MANAGEMENT_MALLOC_REPLY,
	                                 arg0,
	                                 arg1,
	                                 get_low(returnee),
	                                 get_high(returnee)
	                                )
	           )
}

void remote_malloc_real_reply(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                                    gasnet_handlerarg_t arg1,
                                                    gasnet_handlerarg_t arg2,
                                                    gasnet_handlerarg_t arg3
                             ) {
	using namespace adabs::tools;
	long* local   = get_ptr<long> (arg0, arg1);
	void* remote  = get_ptr<void> (arg2, arg3);
	
	*local = reinterpret_cast<long>(remote);
}

void remote_free_real(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                            gasnet_handlerarg_t arg1
                     ) {
	using namespace adabs::tools;
	void* ptr = get_ptr<void>(arg0, arg1);
	free (ptr);
}

}

}
}
#endif
