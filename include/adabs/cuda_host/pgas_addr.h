#pragma once

#include <cuda.h>

#include <utility>
#include <cassert>

#include "adabs/adabs.h"
#include "adabs/cuda_host/allocator.h"
#include "adabs/tools/ptr_divider.h"
#include "adabs/tools/alignment.h"

namespace adabs {

namespace cuda_host {

template <typename T>
struct allocator;

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
				cudaFreeHost(get_flag());
		}
	
	/***************** FUNCTIONS *********************/
	public:
		T* get_data() const {
			using namespace adabs::tools;
			
			// must be called befor is_available is called
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
			__sync_synchronize(); // make sure all data is visible before setting the flag
			int val = __sync_lock_test_and_set(ptr, FULL);
			//std::cout << me << ": avail 2 flag " << *ptr << std::endl;
			return (val == WRITTING || val == REQUESTED);
		}
		
	/****************** FRIEND CLASS **********************/
	friend class allocator<T>;
};



}

}
