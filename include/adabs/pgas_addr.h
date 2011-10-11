#pragma once

#include <utility>
#include <cassert>

#include "adabs/adabs.h"
#include "adabs/allocator.h"
#include "adabs/tools/ptr_divider.h"
#include "adabs/tools/alignment.h"

namespace adabs {

template <typename T>
struct allocator;

namespace pgas {
inline void pgas_addr_remote_get (gasnet_token_t token,
                                  gasnet_handlerarg_t arg0, // data pointer
                                  gasnet_handlerarg_t arg1, // data pointer
                                  gasnet_handlerarg_t arg2, // batch_mem size
                                  gasnet_handlerarg_t arg3, // return data pointer
                                  gasnet_handlerarg_t arg4, // return data pointer
                                  gasnet_handlerarg_t arg5  // flag diff for remote pointer
                                 );
inline void pgas_addr_remote_set (gasnet_token_t token, void *buf, size_t nbytes,
                                  gasnet_handlerarg_t arg0, // data pointer
                                  gasnet_handlerarg_t arg1  // data pointer
                                 );
inline void pgas_addr_set_uninit(gasnet_token_t token,
                                  gasnet_handlerarg_t arg0, // flag pointer
                                  gasnet_handlerarg_t arg1, // flag pointer
                                  gasnet_handlerarg_t arg2, // stride between flags
                                  gasnet_handlerarg_t arg3, // nb of flags
                                  gasnet_handlerarg_t arg4, // done marker pointer
                                  gasnet_handlerarg_t arg5  // done marker pointer
                                );
inline void pgas_addr_check_get_all(gasnet_token_t token,
                                  gasnet_handlerarg_t arg0, // flag pointer
                                  gasnet_handlerarg_t arg1, // flag pointer
                                  gasnet_handlerarg_t arg2, // stride between flags
                                  gasnet_handlerarg_t arg3, // nb of flags
                                  gasnet_handlerarg_t arg4, // done marker pointer
                                  gasnet_handlerarg_t arg5  // done marker pointer
                                );
// TODO move in different file (incl. implementation)
inline void done_marker(gasnet_token_t token,
                                  gasnet_handlerarg_t arg0, // done marker pointer
                                  gasnet_handlerarg_t arg1  // done marker pointer
                                );
}
/**
 * A address in our pgas world
 */
template <typename T>
class pgas_addr {
	/******************* TYPEDEFS ******************/
	public:
		typedef T value_type;
		enum {EMPTY, WRITTING, REQUESTED, FULL};
	
	/******************* VARIABLES *****************/
	private:
		int _orig_node;
		int _batch_size;
		void* _orig_ptr;
		mutable T* _cache;
		
	/**************** CON/DESTRUCTORS ***************/
	public:
		pgas_addr (void* ptr, const int batch_size) : _orig_ptr(ptr),
		                                              _orig_node(adabs::me),
		                                              _batch_size(batch_size),
		                                              _cache(0) {
			if (is_local()) {
				set_cache();
			}
		}
		
		pgas_addr (void* ptr, const int batch_size, const int orig_node)
		                                            : _orig_ptr(ptr),
		                                              _orig_node(orig_node),
		                                              _batch_size(batch_size),
		                                              _cache(0) {
			if (is_local())
				set_cache();
		}
		
		pgas_addr (const pgas_addr<T> &copy) : _orig_ptr(copy._orig_ptr),
		                                 _orig_node(copy._orig_node),
		                                 _batch_size(copy._batch_size),
		                                 _cache(0) {
			if (is_local())
				set_cache();
		}
		
		~pgas_addr() {
			if (!is_local() && _cache!=0)
				free(get_flag());
		}
	
	/***************** FUNCTIONS *********************/
	public:
		T* get_data() const {
			using namespace adabs::tools;
			
			// must be called before is_available is called
			set_cache();
			
			if (!is_local()) {
				const bool r = request();
				
				if (r) { 
					int a = tools::alignment<T>::val();
					if (a<sizeof(int)) a = sizeof(int);
					
					GASNET_CALL(gasnet_AMRequestShort6(_orig_node,
											           adabs::impl::PGAS_ADDR_GET,
											           get_low(_orig_ptr),
											           get_high(_orig_ptr),
											           _batch_size * sizeof(T),
											           get_low(get_flag()),
											           get_high(get_flag()),
											           a
											          )
					          )
				}
			}
			
			while (!is_available()) {
				// wait until data is ready
				__sync_synchronize();
			}
			
			return _cache;
		}
		
		T* get_data_unitialized() {
			set_cache();
			
			const bool w = writing();
			assert(w);
			
			return _cache;
		}
		
		void set_data(T const * const data) {
			assert (data == _cache);
			
			//std::cout << me << ": fill data in " << _cache << " is local " << is_local() << std::endl;
			
			using namespace adabs::tools;
			
			const bool a = available();
			assert (a);
			if (!is_local()) {
				GASNET_CALL(gasnet_AMRequestLong2(_orig_node,
										           adabs::impl::PGAS_ADDR_SET,
										           _cache,
										           sizeof(T)*_batch_size,
										           (void*)((int*)_orig_ptr + 1),
										           get_low(_orig_ptr),
										           get_high(_orig_ptr)
										          )
				          )
			}
		}

		bool is_local() const {
			return (_orig_node == adabs::me);
		}
		
		pgas_addr<T>& operator+= (const int rhs) {
			int a = tools::alignment<T>::val();
			if (a<sizeof(int)) a = sizeof(int);
			
			_orig_ptr = (void*)(
			             (char*)_orig_ptr
			             + (a + sizeof(T)*_batch_size)*rhs
			            );
			return *this;
		}
		
		pgas_addr<T> operator+(const int rhs) const {
			int a = tools::alignment<T>::val();
			if (a<sizeof(int)) a = sizeof(int);
			
			char* add = (char*)_orig_ptr;
			add += (a + sizeof(T)*_batch_size)*rhs;
			
			return pgas_addr<T>(add, _batch_size, _orig_node);
		}
		
		pgas_addr<T>& operator-= (const int rhs) {
			int a = tools::alignment<T>::val();
			if (a<sizeof(int)) a = sizeof(int);
			
			_orig_ptr = (void*)(
			             (char*)_orig_ptr
			             - (a + sizeof(T)*_batch_size)*rhs
			            );
			return *this;
		}
		
		pgas_addr<T> operator-(const int rhs) const {
			int a = tools::alignment<T>::val();
			if (a<sizeof(int)) a = sizeof(int);
			
			char* add = (char*)_orig_ptr;
			add -= (a + sizeof(T)*_batch_size)*rhs;
			
			return pgas_addr<T>(add, _batch_size, _orig_node);
		}
		
		/**
		 * This function call is unsafe, only use with care
		 * We'll have to check on which occasions we use it and than
		 * offer better solutions for these.
		 */
		void* get_raw_pointer() const {
			return _orig_ptr;
		}
		
		int get_node() const {
			return _orig_node;
		}
		
		int get_batch_size() const {
			return _batch_size;
		}
		
		int* get_orig_flag() const {
			//std::cout << "orig_ptr = " << _orig_ptr << std::endl;
			return (int*)(_orig_ptr);
		}
		
		int* get_flag() const {
			assert(_cache!=0);
			int a = tools::alignment<T>::val();
			if (a<sizeof(int)) a = sizeof(int);
			
			//std::cout << "flag is at " << (int*)((char*)_cache - a) << std::endl;
			return (int*)((char*)_cache - a);
		}
		
	private:
		void set_cache() const {
			if (is_local()) {
				int a = tools::alignment<T>::val();
				if (a<sizeof(int)) a = sizeof(int);
				_cache = (T*)((char*)(_orig_ptr)+a);
			} else {
				#pragma omp critical
				{
					if (_cache == 0)
						_cache = allocator<T>::allocate(_batch_size, _batch_size).get_data_ptr();
				}
				
			}
		}
		
		T* get_data_ptr() const {
			return _cache;
		}
		
		// check if flag is set to 1
		bool is_writing() const {
			volatile int *reader = get_flag();
			return (*reader == WRITTING);
		}
		
		// check if flag is set to 2
		bool is_requested() const {
			volatile int *reader = get_flag();
			return (*reader == REQUESTED);
		}
		
		// check if flag is set to 3
		bool is_available() const {
			volatile int *reader = get_flag();
			return (*reader == FULL);
		}
		
		// check and set flag to 1
		bool writing() const {
			volatile int *ptr = get_flag();
			//std::cout << me << ": writting flag " << get_flag() << " - " << *ptr << std::endl;
			return __sync_bool_compare_and_swap(ptr, EMPTY, WRITTING);
		}
		
		// check and set flag to 2
		bool request() const {
			volatile int *ptr = get_flag();
			//std::cout << me << ": request flag " << get_flag() << " - " << *ptr << " set to 2 " << std::endl; 
			return __sync_bool_compare_and_swap(ptr, EMPTY, REQUESTED);
		}
		
		// check and set flag to 3
		bool available() const {
			volatile int *ptr = get_flag();
			//std::cout << me << ": avail 1 flag " << get_flag() << " - " << *ptr << std::endl;
			int val = __sync_lock_test_and_set(ptr, FULL);
			//std::cout << me << ": avail 2 flag " << *ptr << std::endl;
			return (val == WRITTING || val == REQUESTED);
		}
		
	/****************** FRIEND CLASS **********************/
	friend class allocator<T>;
};

namespace pgas {

inline void pgas_addr_set_uninit(gasnet_token_t token,
                                  gasnet_handlerarg_t arg0, // flag pointer
                                  gasnet_handlerarg_t arg1, // flag pointer
                                  gasnet_handlerarg_t arg2, // stride between flags
                                  gasnet_handlerarg_t arg3, // nb of flags
                                  gasnet_handlerarg_t arg4, // done marker pointer
                                  gasnet_handlerarg_t arg5  // done marker pointer
                                ) {
	using namespace adabs::tools;
	int *flag  = get_ptr<int>(arg0, arg1);
	
	//std::cout << "flag should be at " << flag << std::endl;
	
	for (int i=0; i<arg3; ++i) {
		int writting = __sync_val_compare_and_swap(flag, adabs::pgas_addr<void>::EMPTY, adabs::pgas_addr<void>::WRITTING);
		assert(writting==adabs::pgas_addr<void>::EMPTY);
		flag = (int*)((char*)flag + arg2);
	}
	
	int *return_marker  = get_ptr<int>(arg4, arg5);
	if (return_marker != 0) {
	GASNET_CALL(gasnet_AMReplyShort2(token,
								       adabs::impl::SET_RETURN_MARKER,
								       arg4,
								       arg5
								      )
		      )
	}
}

/**
 * Pthread argument class
 */
struct pgas_addr_check_get_all_thread_arg {
	volatile int *flag;
	gasnet_handlerarg_t arg2;
	gasnet_handlerarg_t arg3;
	gasnet_handlerarg_t arg4;
	gasnet_handlerarg_t arg5;
	gasnet_node_t dest;
	
	
	pgas_addr_check_get_all_thread_arg(volatile int *_flag, 
	                                   gasnet_handlerarg_t _arg2,
	                                   gasnet_handlerarg_t _arg3,
	                                   gasnet_handlerarg_t _arg4,
	                                   gasnet_handlerarg_t _arg5,
	                                   gasnet_node_t _dest
	                                  ) : flag(_flag),
	                                      arg2(_arg2),
	                                      arg3(_arg3),
	                                      arg4(_arg4),
	                                      arg5(_arg5),
	                                      dest(_dest) {}
};

inline void* pgas_addr_check_get_all_thread(void *threadarg) {
	using namespace adabs::tools;
	
	pgas_addr_check_get_all_thread_arg* arg = (pgas_addr_check_get_all_thread_arg*)threadarg;
	
	for (int i=0; i<arg->arg3; ++i) {
		std::cout << i << " at " << (int*)arg->flag << std::endl;
		
		// wait until flag is set
		volatile int* reader = arg->flag;
		while (*reader != adabs::pgas_addr<void>::FULL) {}
		
		arg->flag = (volatile int*)((char*)arg->flag + arg->arg2);
	}
	
	int *return_marker  = get_ptr<int>(arg->arg4, arg->arg5);
	std::cout << "set done flag at " << return_marker << std::endl;
	if (return_marker != 0) {
		GASNET_CALL(gasnet_AMRequestShort2(arg->dest,
										   adabs::impl::SET_RETURN_MARKER,
										   arg->arg4,
										   arg->arg5
										  )
				  )
	}
	           
	delete arg;
	
	pthread_exit(0);
}

inline void pgas_addr_check_get_all(gasnet_token_t token,
                                  gasnet_handlerarg_t arg0, // flag pointer
                                  gasnet_handlerarg_t arg1, // flag pointer
                                  gasnet_handlerarg_t arg2, // stride between flags
                                  gasnet_handlerarg_t arg3, // nb of flags
                                  gasnet_handlerarg_t arg4, // done marker pointer
                                  gasnet_handlerarg_t arg5  // done marker pointer
                                ) {
	using namespace adabs::tools;
	volatile int *flag  = get_ptr<volatile int>(arg0, arg1);
	gasnet_node_t src;
	GASNET_CALL( gasnet_AMGetMsgSource(token, &src) )
	
	pgas_addr_check_get_all_thread_arg *para = new pgas_addr_check_get_all_thread_arg(flag, arg2, arg3, arg4, arg5, src);
	
	pthread_t thread_id;
	pthread_attr_t attrb;
	pthread_attr_init(&attrb);
	pthread_attr_setdetachstate(&attrb, PTHREAD_CREATE_DETACHED);
	pthread_create(&thread_id, &attrb, pgas_addr_check_get_all_thread, (void*) para);

}

inline void done_marker(gasnet_token_t token,
                                  gasnet_handlerarg_t arg0, // done marker pointer
                                  gasnet_handlerarg_t arg1  // done marker pointer
                                ) {
	using namespace adabs::tools;
	int *return_marker  = get_ptr<int>(arg0, arg1);
	int val = __sync_lock_test_and_set(return_marker, 1);
	//std::cout << "changed " << return_marker << std::endl;
	assert (val == 0);
}


inline void pgas_addr_remote_set (gasnet_token_t token, void *buf, size_t nbytes,
                                  gasnet_handlerarg_t arg0, // data pointer
                                  gasnet_handlerarg_t arg1  // data pointer
                                 ) {
	using namespace adabs::tools;
	
	int *flag  = get_ptr<int>(arg0, arg1);
	
	//std::cout << gasnet_mynode() << " remote set receviced " << flag << std::endl;
	
	__sync_synchronize();
	
	int val = __sync_lock_test_and_set(flag, adabs::pgas_addr<void>::FULL);
	assert(val != adabs::pgas_addr<void>::FULL);
}


/**
 * Pthread argument class
 */
struct remote_get_thread_arg {
	void *local;
	const int batch_mem_size;
	void *remote;
	const gasnet_node_t dest;
	const int flag_diff;
	
	remote_get_thread_arg(void *_local, 
	                      const int _batch_mem_size,
	                      void *_remote,
	                      gasnet_node_t _dest,
	                      const int _flag_diff
	                      ) : local(_local),
	                          batch_mem_size(_batch_mem_size),
	                          remote(_remote),
	                          dest(_dest),
	                          flag_diff(_flag_diff) {}
};

inline void* remote_get_thread(void *threadarg) {
	using namespace adabs::tools;
	
	remote_get_thread_arg* arg = (remote_get_thread_arg*)threadarg;
	
	// wait until flag is set
	volatile int* reader = (volatile int*)arg->local;
	//std::cout << me << ": " << arg->local << std::endl;
	//std::cout << me << ": " << *reader << std::endl;
	while (*reader != adabs::pgas_addr<void>::FULL) {}
	
	__sync_synchronize();
	
	// sent data back
	void* buf = (char*)arg->local + arg->flag_diff;

	int* data = (int*)arg->remote; // flag
	++data; // move to data

	GASNET_CALL(
	            gasnet_AMRequestLong2(arg->dest, adabs::impl::PGAS_ADDR_SET,
	                                  buf, arg->batch_mem_size, data,
	                                  get_low(arg->remote),
	                                  get_high(arg->remote)
	                                 )
	           )
	           
	delete arg;
	
	//std::cout << "remote get completed" << std::endl;
	
	pthread_exit(0);
}

inline void pgas_addr_remote_get (gasnet_token_t token,
                                  gasnet_handlerarg_t arg0, // data pointer
                                  gasnet_handlerarg_t arg1, // data pointer
                                  gasnet_handlerarg_t arg2, // batch_mem size
                                  gasnet_handlerarg_t arg3, // return data pointer
                                  gasnet_handlerarg_t arg4, // return data pointer
                                  gasnet_handlerarg_t arg5  // flag diff for remote pointer
                                 ) {
	using namespace adabs::tools;
	
	//std::cout << gasnet_mynode() << " remote get receviced" << std::endl;
	
	void *local  = get_ptr<void>(arg0, arg1);
	void *remote = get_ptr<void>(arg3, arg4);
	
	gasnet_node_t src;
	GASNET_CALL( gasnet_AMGetMsgSource(token, &src) )
	
	remote_get_thread_arg *para = new remote_get_thread_arg(local, arg2, remote, src, arg5);
	
	pthread_t thread_id;
	pthread_attr_t attrb;
	pthread_attr_init(&attrb);
	pthread_attr_setdetachstate(&attrb, PTHREAD_CREATE_DETACHED);
	pthread_create(&thread_id, &attrb, remote_get_thread, (void*) para);
}

}



}
