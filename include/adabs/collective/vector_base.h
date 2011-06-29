#pragma once

#include "adabs/gasnet_config.h"

#include "adabs/adabs.h"
#include "adabs/tools/tools.h"
#include "adabs/tools/ptr_divider.h"

namespace adabs {

namespace collective {

class vector_base {
		/************************ TYPEDEFS ***************************/
		
	
		/************************ VARIABLES **************************/
	public:
		vector_base** _proxies; // TODO make private
		
	public:
		/**
		 * Do not touch these variable unless you _really_ know how it is used!
		 */
		static vector_base** global_com;
				
		/********************* CON/DESTRUCTOS ************************/
	private:
		
	public:
		vector_base() {
			using namespace adabs::tools;
	
			// call set_global_com on all nodes
			const int all = gasnet_nodes();
			for (int i=0; i<all; ++i) {
					GASNET_CALL(gasnet_AMRequestShort2(i,
							                           adabs::impl::COLLECTIVE_VECTOR_GLOBAL_COM_SET,
							                           get_low(this),
							                           get_high(this)
							                          )
							   )
			}
	
			for (int i=0; i<all; ++i) {
				GASNET_BLOCKUNTIL(vector_base::global_com[i]!=0);
			}
	
			_proxies = new vector_base*[all];
			for (int i=0; i<all; ++i) {
				_proxies[i] = global_com[i];
				global_com[i] = 0;
			}
	

		}
		
		vector_base (const vector_base& cpy) {
			const int all = gasnet_nodes();

			_proxies = new vector_base*[all];
			for (int i=0; i<all; ++i) {
				_proxies[i] = cpy._proxies[i];
			}
		}
		
		~vector_base() {}
		
		/************************ FUNCTIONS **************************/
	public:
		virtual void set(const int x, const void* const ptr) = 0; // should define pgas friends for this function
};


namespace pgas {

inline void set_global_com (gasnet_token_t token, gasnet_handlerarg_t arg0,
                                           gasnet_handlerarg_t arg1) {
	using namespace adabs::tools;
	
	gasnet_node_t source;
	vector_base * ptr = get_ptr<vector_base>(arg0, arg1);
	
	GASNET_CALL(gasnet_AMGetMsgSource(token, &source))
	
	vector_base::global_com[source] = ptr;
}

inline void remote_set_vector_element (gasnet_token_t token, void *buf, size_t nbytes,
                                       gasnet_handlerarg_t arg0,
                                       gasnet_handlerarg_t arg1,
                                       gasnet_handlerarg_t arg2) {
	using namespace adabs::tools;
	
	const int i = arg2;
	gasnet_node_t source;
	vector_base * that = get_ptr<vector_base>(arg0, arg1);
	
	that -> set(i, buf);
}


}

}

}
