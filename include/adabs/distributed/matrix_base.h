#pragma once

#include "adabs/tools/ptr_divider.h"

namespace adabs {

namespace distributed {


class matrix_base {
		int delete_flag;
		int use_flag;
		int reuse_all_remote_counter;

	public:
		virtual bool remove(const bool local)=0;
		
		void set_use_flag_start () {
			__sync_lock_test_and_set (&use_flag, 0);
		}
		
		int get_delete_flag() const {
			const volatile int *ptr = &delete_flag;
			return *ptr;
		}
		
		void set_delete_flag(const int par) {
			__sync_lock_test_and_set (&delete_flag, par);
		}
		
		void use() {
			__sync_lock_test_and_set (&use_flag, 1);
		}
		
		void set_use_flag(const int par) {
			__sync_lock_test_and_set (&use_flag, par);
		}
		
		void wait_for_reuse() const {
			const volatile int *ptr = &use_flag;
			
			while (*ptr != 0) {}
		}
		
		bool resetted() {
			volatile int *ptr = &use_flag;
			return (*ptr == 2) ||  (*ptr == 3);
		}
		
		bool all_resetted() {
			volatile int *ptr = &use_flag;
			return *ptr == 3;
		}
		
		void reuse_all_remote_done() {
			#pragma omp atomic
			++reuse_all_remote_counter;
		}
		
		void reset_reuse_all_remote_counter() {
			__sync_lock_test_and_set (&reuse_all_remote_counter, 0);
		}
		
		void wait_until_remote_reuse_all() {
			const int all = gasnet_nodes()-1;
			const volatile int *ptr = &reuse_all_remote_counter;
			
			while (*ptr != all) {}
		}
		
		virtual bool enable_reuse(const bool)=0;
		virtual void enable_reuse_all(const bool first_caller)=0;
		
		matrix_base() : delete_flag(0), use_flag(0), reuse_all_remote_counter(0) {}
		
		virtual ~matrix_base() {};
};

namespace pgas {

inline void delete_matrix(gasnet_token_t token, gasnet_handlerarg_t arg0, gasnet_handlerarg_t arg1) {
	using namespace adabs::tools;
	
	matrix_base * const that = get_ptr<matrix_base>(arg0, arg1);
	
	delete that;
}

inline void remove_matrix(gasnet_token_t token, gasnet_handlerarg_t arg0, gasnet_handlerarg_t arg1, gasnet_handlerarg_t arg2, gasnet_handlerarg_t arg3) {
	using namespace adabs::tools;
	
	
	matrix_base * const that = get_ptr<matrix_base>(arg0, arg1);
	
	bool all_removed = that->remove(false);
	
	if (all_removed) {
		GASNET_CALL(
			        gasnet_AMReplyShort3(token, adabs::impl::DISTRIBUTED_MATRIX_BASE_DELETE_ALL, arg2, arg3, 2)
			       )
	} else {
		GASNET_CALL(
			        gasnet_AMReplyShort3(token, adabs::impl::DISTRIBUTED_MATRIX_BASE_DELETE_ALL, arg2, arg3, 1)
			       )
	}
}

inline void delete_all_matrix(gasnet_token_t token, gasnet_handlerarg_t arg0, gasnet_handlerarg_t arg1, gasnet_handlerarg_t arg2) {
	using namespace adabs::tools;
	
	matrix_base * const that = get_ptr<matrix_base>(arg0, arg1);
	
	that -> set_delete_flag(arg2);
}


inline void reuse_matrix_reply(gasnet_token_t token, gasnet_handlerarg_t arg0, gasnet_handlerarg_t arg1, gasnet_handlerarg_t arg2) {
	using namespace adabs::tools;
	
	matrix_base * const that = get_ptr<matrix_base>(arg0, arg1);
	
	that -> set_use_flag(arg2);
}
 

inline void reuse_matrix(gasnet_token_t token, gasnet_handlerarg_t arg0, gasnet_handlerarg_t arg1, gasnet_handlerarg_t arg2, gasnet_handlerarg_t arg3) {
	using namespace adabs::tools;
	
	matrix_base * const that = get_ptr<matrix_base>(arg0, arg1);
	
	const bool all_reuse = that -> enable_reuse(false);
	
	if (all_reuse) {
		// everyone else should reset their data
		GASNET_CALL(
			        gasnet_AMReplyShort3(token, adabs::impl::DISTRIBUTED_MATRIX_BASE_REUSE_REPLY, arg2, arg3, 3)
			       )
	} else {
		// ok, not everyone is ready
		GASNET_CALL(
			        gasnet_AMReplyShort3(token, adabs::impl::DISTRIBUTED_MATRIX_BASE_REUSE_REPLY, arg2, arg3, 2)
			       )
	}
}

inline void reuse_all_matrix(gasnet_token_t token, gasnet_handlerarg_t arg0, gasnet_handlerarg_t arg1, gasnet_handlerarg_t arg2, gasnet_handlerarg_t arg3) {
	using namespace adabs::tools;
	
	matrix_base * const that = get_ptr<matrix_base>(arg0, arg1);
	
	that -> enable_reuse_all(false);
	
	GASNET_CALL(
		        gasnet_AMReplyShort2(token, adabs::impl::DISTRIBUTED_MATRIX_BASE_ENABLE_REUSE_ALL_REPLY, arg2, arg3)
		       )
}

inline void reuse_all_matrix_reply(gasnet_token_t token, gasnet_handlerarg_t arg0, gasnet_handlerarg_t arg1) {
	using namespace adabs::tools;
	
	matrix_base * const that = get_ptr<matrix_base>(arg0, arg1);
	
	that -> reuse_all_remote_done();	
}

inline void remote_reset_reuse_all_remote_counter(gasnet_token_t token, gasnet_handlerarg_t arg0, gasnet_handlerarg_t arg1) {
	using namespace adabs::tools;
	
	matrix_base * const that = get_ptr<matrix_base>(arg0, arg1);
	
	that -> set_use_flag_start();	
}

}

}

}
