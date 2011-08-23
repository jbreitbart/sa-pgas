#pragma once

#include <iostream>
#include <pthread.h>

#include "adabs.h"
#include "pgas_addr.h"

#include "adabs/impl/callbacks.h"

#include "tools/tile.h"
#include "tools/ptr_divider.h"

namespace adabs {

/**
 * This is a matrix base class used for both remote matrix and local
 * matrix. This class is required for the PGAS communication.
 * 
 * At the bottom of the file are also some non class functions, which
 * should be friend functions (and some data/functions of the class
 * should be private).
 */
class matrix_base {
		/************************ VARIABLES **************************/
	private:
		int _size_x;
		int _size_y;
		void* _pgas_data;
		
		/********************* CON/DESTRUCTOS ************************/
	private:
		
	public:
		matrix_base (const int size_x, const int size_y) : _size_x(size_x), _size_y(size_y) {
		}
		
		matrix_base (const matrix_base& cpy) : _size_x(cpy._size_x), _size_y(cpy._size_y) {
		}

		/**
		 * Make sure thet @param T is a matrix_base (or a class inheriated from matrix_base)
		 */
		template <typename T>
		matrix_base (const pgas_addr<T>& ptr) : _size_x(-1), _size_y(-1) {
			using namespace adabs::tools;
			
			GASNET_CALL(gasnet_AMRequestShort4(ptr.get_node(),
			                                   adabs::impl::MATRIX_BASE_INIT_GET,
			                                   get_low(ptr.get_ptr()),
			                                   get_high(ptr.get_ptr()),
			                                   get_low(this),
			                                   get_high(this)
			                                  )
			           )
			volatile int *reader = &_size_y;
			while (*reader==-1) {
				
			}
			
			//GASNET_BLOCKUNTIL(_size_y!=-1);
		}
		
		~matrix_base() {}
		
		/************************ FUNCTIONS **************************/
	private:

	public:
		
		/**
		 * Returns the size in x dimension
		 */
		int get_size_x() const {return _size_x;}
		
		/**
		 * Returns the size in x dimension
		 */
		int get_size_y() const {return _size_y;}
		
		/**
		 * An atomic set instruction. Will only work once on a matrix.
		 */
		void set_size_x(const int x) {
			__sync_lock_test_and_set(&_size_x, x);
		}
		
		/**
		 * An atomic set instruction. Will only work once on a matrix.
		 */
		void set_size_y(const int y) {
			__sync_lock_test_and_set(&_size_y, y);
		}
		
		virtual int pgas_tile_size() const = 0;
		virtual void* pgas_get(const int x, const int y) const = 0;
		virtual void  pgas_mark(const int x, const int y) = 0;
		
		void* pgas_get_data_ptr() const {
			return _pgas_data;
		}
		void  pgas_set_data_ptr(void* ptr) {
			_pgas_data = ptr;
		}
};

namespace pgas {

inline void matrix_init_get(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                                     gasnet_handlerarg_t arg1,
                                                     gasnet_handlerarg_t arg2,
                                                     gasnet_handlerarg_t arg3
                              ) {
	using namespace adabs::tools;
	
	const matrix_base * const that   = get_ptr<const matrix_base>(arg0, arg1);
	const matrix_base *       source = get_ptr<const matrix_base>(arg2, arg3);

	const int x = that->get_size_x();
	const int y = that->get_size_y();
	const void* data_ptr = that->pgas_get_data_ptr();
	
	// TODO check if we really need data_ptr + the original pointer
	__sync_synchronize();
	GASNET_CALL(gasnet_AMReplyShort6(token, adabs::impl::MATRIX_BASE_INIT_SET,
	                                 get_low(source),
	                                 get_high(source),
	                                 get_low(data_ptr),
	                                 get_high(data_ptr),
	                                 x, y
	                                )
	           )
}


inline void matrix_init_set(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                                  gasnet_handlerarg_t arg1,
                                                  gasnet_handlerarg_t arg2,
                                                  gasnet_handlerarg_t arg3,
                                                  gasnet_handlerarg_t arg4,
                                                  gasnet_handlerarg_t arg5
                           ) {
                           
	using namespace adabs::tools;
	matrix_base * const that = get_ptr<matrix_base>(arg0, arg1);
	void * data_ptr = get_ptr<void>(arg2, arg3);
	const int x = (int)arg4;
	const int y = (int)arg5;
		
	that->set_size_x(x);
	that->set_size_y(y);
	that->pgas_set_data_ptr(data_ptr);
	__sync_synchronize();
}

inline void remote_set_matrix_tile (gasnet_token_t token, void *buf, size_t nbytes,
                                       gasnet_handlerarg_t arg0,
                                       gasnet_handlerarg_t arg1,
                                       gasnet_handlerarg_t arg2,
                                       gasnet_handlerarg_t arg3) {
	using namespace adabs::tools;
	
	const int x = arg2;
	const int y = arg3;
	matrix_base * that = get_ptr<matrix_base>(arg0, arg1);
	
	that -> pgas_mark (x, y);
	__sync_synchronize();
}

/**
 * Pthread argument class
 */
struct remote_get_matrix_tile_thread_arg {
	matrix_base *that;
	int x;
	int y;
	gasnet_handlerarg_t arg2;
	gasnet_handlerarg_t arg3;
	void *remote_data_ptr;
	gasnet_node_t dest;
	
	remote_get_matrix_tile_thread_arg(matrix_base *_that,
	                                  const int _x, const int _y,
	                                  gasnet_handlerarg_t _arg2, gasnet_handlerarg_t _arg3,
	                                  void* _remote_data_ptr,
	                                  gasnet_node_t _dest
	                                 ) {
		that = _that;
		x = _x;
		y = _y;
		arg2 = _arg2;
		arg3 = _arg3;
		remote_data_ptr = _remote_data_ptr;
		dest = _dest;
		
	}
};

inline void* remote_get_matrix_tile_thread(void *threadarg) {
	remote_get_matrix_tile_thread_arg* arg = (remote_get_matrix_tile_thread_arg*)threadarg;
	
	void* buf = arg->that -> pgas_get(arg->x, arg->y);
	const int tile_size = arg->that->pgas_tile_size();

	GASNET_CALL(
	            gasnet_AMRequestLong4(arg->dest, adabs::impl::MATRIX_BASE_SET,
	                                  buf, tile_size, arg->remote_data_ptr, arg->arg2, arg->arg3, arg->x, arg->y)
	           )
	           
	delete arg;
	
	
	pthread_exit(0);
}

inline void remote_get_matrix_tile (gasnet_token_t token,
                                       gasnet_handlerarg_t arg0,
                                       gasnet_handlerarg_t arg1,
                                       gasnet_handlerarg_t arg2,
                                       gasnet_handlerarg_t arg3,
                                       gasnet_handlerarg_t arg4,
                                       gasnet_handlerarg_t arg5,
                                       gasnet_handlerarg_t arg6,
                                       gasnet_handlerarg_t arg7) {
	using namespace adabs::tools;
	
	//std::cout << gasnet_mynode() << " remote get receviced" << std::endl;
	
	matrix_base * that = get_ptr<matrix_base>(arg0, arg1);
	void *remote_data_ptr = get_ptr<void*>(arg4, arg5);

	const int x = arg6;
	const int y = arg7;
	
	gasnet_node_t src;
	GASNET_CALL( gasnet_AMGetMsgSource(token, &src) )
	
	remote_get_matrix_tile_thread_arg *para =
	           new remote_get_matrix_tile_thread_arg(that, x, y, arg2, arg3, remote_data_ptr, src);
	
	pthread_t thread_id;
	pthread_attr_t attrb;
	pthread_attr_init(&attrb);
	pthread_attr_setdetachstate(&attrb, PTHREAD_CREATE_DETACHED);
	pthread_create(&thread_id, &attrb, remote_get_matrix_tile_thread, (void*) para);
	
#if 0
	//std::cout << gasnet_mynode() << " remote get pgas_get will be called" << std::endl;
	//void* buf = that -> pgas_get(x, y);
	//const int tile_size = that->pgas_tile_size();
	
	//std::cout << gasnet_mynode() << " remote get completed" << std::endl;
	//GASNET_CALL(
	//            gasnet_AMReplyLong4(token, adabs::impl::MATRIX_BASE_SET,
	//                                  buf, tile_size, remote_data_ptr, arg2, arg3, arg6, arg7)
	//           )
#endif
}

}
}
