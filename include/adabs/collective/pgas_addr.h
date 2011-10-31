#pragma once

#include <utility>
#include <cassert>

#include "adabs/adabs.h"
#include "adabs/collective/allocator.h"
#include "adabs/tools/ptr_divider.h"
#include "adabs/tools/alignment.h"

namespace adabs {
namespace collective {

template <typename T>
struct allocator;

namespace pgas {
inline void pgas_addr_remote_set (gasnet_token_t token, void *buf, size_t nbytes,
                                  gasnet_handlerarg_t arg0, // data pointer
                                  gasnet_handlerarg_t arg1  // data pointer
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
	
	/******************* VARIABLES *****************/
	private:
		const int _batch_size;
		void* _orig_ptr;
		T* _ptr;
		
	/**************** CON/DESTRUCTORS ***************/
	public:
		pgas_addr (void* ptr, const int batch_size) : _orig_ptr(ptr),
		                                              _batch_size(batch_size) {
			int a = tools::alignment<T>::val();
			if (a<sizeof(int)) a = sizeof(int);
			
			const int pointer_alignment = a - ((adabs::all*sizeof(void*)) % a);
		    _ptr = (T*)((char*)_orig_ptr + sizeof(void*)*adabs::all + pointer_alignment);
		    //std::cout << me << ": " << "my data lifes on " << _ptr << std::endl;
		    
		}
		
		pgas_addr (const pgas_addr<T> &copy) : _orig_ptr(copy._orig_ptr),
		                                       _batch_size(copy._batch_size),
		                                       _ptr(copy._ptr) {}
		
		~pgas_addr() {}
	
	/***************** FUNCTIONS *********************/
	public:
		T* get_data() const {
			
			while (!is_available()) {
				// wait until data is ready
			}
			
			return _ptr;
		}
		
		T* get_data_unitialized() {
			const bool w = writing();
			assert(w);
			
			return _ptr;
		}
		
		void set_data(T const * const data) {
			assert (data == _ptr);
			
			using namespace adabs::tools;
			
			__sync_synchronize();
			
			const bool a = available();
			assert (a);
			
			for (int i=0; i<adabs::all; ++i) {
				if (i == adabs::me) continue;
				
				void* remote_ptr = get_remote_ptr(i);
				void* remote_flag = get_remote_flag(i);
				
				GASNET_CALL(gasnet_AMRequestLong2(i,
											      adabs::impl::COLLECTIVE_PGAS_ADDR_SET,
											      _ptr,
											      sizeof(T)*_batch_size,
											      remote_ptr,
											      get_low(remote_flag),
											      get_high(remote_flag)
											     )
					      )
			}
		}

		bool is_local() const {
			return true;
		}
		
		pgas_addr<T>& operator+= (const int rhs) {
			int a = tools::alignment<T>::val();
			if (a<sizeof(int)) a = sizeof(int);
			_ptr = (T*)((char*)_ptr
			               + (a + sizeof(T)*_batch_size)*rhs
			            );
			return *this;
		}
		
		pgas_addr<T>& operator-= (const int rhs) {
			int a = tools::alignment<T>::val();
			if (a<sizeof(int)) a = sizeof(int);
			_ptr = (T*)((char*)_ptr
			               - (a + sizeof(T)*_batch_size)*rhs
			              );
			return *this;
		}
		
		pgas_addr<T> operator+(const int rhs) const {
			pgas_addr<T> copy(*this);
			copy += rhs;
			
			return copy;
		}
		
		pgas_addr<T> operator-(const int rhs) const {
			pgas_addr<T> copy(*this);
			copy -= rhs;
			
			return copy;
		}
		
	private:
		
		void* get_remote_orig_ptr(const int node) {
			void* result = *((void**)_orig_ptr + node);
			
			return result;
		}
		
		void* get_remote_ptr(const int node) {
			char* start = (char*)get_remote_orig_ptr(node);
			
		    char* result = ((char*)_ptr - (char*)_orig_ptr) + start;
		    
			return (void*)result;
		}
		
		void* get_remote_flag (const int node) {
			char *ptr = (char*)get_remote_ptr(node);
			ptr += _batch_size * sizeof(T);
			return ptr;
		}
		
		
		int* get_flag() const {
			assert(_ptr!=0);
			
			return (int*)((char*)_ptr + _batch_size*sizeof(T));
		}
		
		T* get_data_ptr() const {
			return _ptr;
		}
		
		// check if flag is set to 1
		bool is_writing() const {
			volatile int *reader = get_flag();
			return (*reader == 1);
		}
		
		// check if flag is set to 3
		bool is_available() const {
			volatile int *reader = get_flag();

			return (*reader == 3);
		}
		
		// check and set flag to 1
		bool writing() const {
			volatile int *ptr = get_flag();
			return __sync_bool_compare_and_swap(ptr, 0, 1);
		}
		

		// check and set flag to 3
		bool available() const {
			volatile int *ptr = get_flag();
			int val = __sync_lock_test_and_set(ptr, 3);
			return (val == 1);
		}
		
	/****************** FRIEND CLASS **********************/
	friend class allocator<T>;
};

namespace pgas {

inline void pgas_addr_remote_set (gasnet_token_t token, void *buf, size_t nbytes,
                                  gasnet_handlerarg_t arg0, // data pointer
                                  gasnet_handlerarg_t arg1  // data pointer
                                 ) {
	using namespace adabs::tools;
	
	int *flag  = get_ptr<int>(arg0, arg1);
	
	__sync_synchronize();
	
	int val = __sync_lock_test_and_set(flag, 3);
	
	assert(val != 3);
}

}



}
}
