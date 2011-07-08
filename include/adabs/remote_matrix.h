#pragma once

#include "adabs/tools/tools.h"
#include "adabs/tools/tile.h"

#include "adabs/matrix_base.h"
#include "adabs/matrix.h"

namespace adabs {

/**
 * This is a tile based remote matrix class.
 *
 * Note: You should not inheriate from this matrix.
 */
template <typename T, int tile_size>
class remote_matrix : public matrix_base {
		/************************ TYPEDEFS ***************************/
	private:
		typedef tools::tile<T, tile_size> tile;
		typedef tile* dataT;
		
		typedef adabs::matrix<T, tile_size> sourceT;
	
		/************************ VARIABLES **************************/
	private:
		dataT *_data;
		const int _nb_tiles_x;
		const int _nb_tiles_y;

		const pgas_addr <sourceT> _addr;
		
		/********************* CON/DESTRUCTOS ************************/
	private:
		
	public:
		
		/**
		 * Copy constructor to create a copy of @param cpy
		 */
		remote_matrix (const pgas_addr <sourceT>& addr) : matrix_base(addr), _addr(addr),
		    _nb_tiles_x((get_size_x()%tile_size == 0) ? (get_size_x()/tile_size) : (get_size_x()/tile_size+1)),
		    _nb_tiles_y((get_size_y()%tile_size == 0) ? (get_size_y()/tile_size) : (get_size_y()/tile_size+1)) {

			const int tiles_y = get_nb_tile_y();
			const int tiles_x = get_nb_tile_x();
			_data = new dataT[tiles_y*tiles_x];
			for (int i=0; i<tiles_y*tiles_x; ++i)
				_data[i] = 0;
		}
		
		/**
		 * Desctructor, make sure to not delete the object before(!) all
		 * reads to that matrix are completed.
		 */
		~remote_matrix() {
			for (int i = 0; i<get_nb_tile_y()*get_nb_tile_x(); ++i) {
				delete _data[i];
			}
			delete []_data;
		}
		
		/************************ FUNCTIONS **************************/
	private:

	public:
		/**
		 * Copies the values stored in
		 * @param ptr[0 ... tile_size*tile_size]
		 * to the tile with the coordinates
		 * @param x and @param y
		 * and marks the values as initialized. If @param *ptr is
		 * identical to a pointer returned by get_tile_unitialized
		 * no data will be copied.
		 */
		void set_tile(T const * restrict const ptr, const int x, const int y, const bool sent=true);
		
		/**
		 * Returns a pointer to the tile with the coordinated
		 * @param x and @param y
		 * . In case the values are not yet written to the matrix, the
		 * calling thread will sleep until the value is returned.
		 */
		T const* get_tile(const int x, const int y);
		
		/**
		 * Returns the pointer to the matrix internal tile with the
		 * coordinates
		 * @param x and  @param y
		 * so one can update the matrix in place. You must(!) still call
		 * set_tile() for this matrix tile!
		 */
		T* get_tile_unitialized(const int x, const int y);
		
		/**
		 * Returns the tile size
		 */
		static int get_tile_size() {return tile_size;}	

		/**
		 * Returns the number of tiles in x-dimension
		 */
		int get_nb_tile_x() const {
			return _nb_tiles_x;
		}
		
		/**
		 * Returns the number of tiles in y-dimension
		 */
		int get_nb_tile_y() const {
			return _nb_tiles_y;
		}

		void* pgas_get(const int x, const int y) const {
			throw "should not be called";
		}
		
		void  pgas_mark(const int x, const int y) {
			if (_data[y*_nb_tiles_x + x] == 0) {
				std::cerr << "Error that should not be 12c" << std::endl;
				exit(-1);
			}
			__sync_lock_test_and_set (&(_data[y*_nb_tiles_x + x]->flag), 2);
		}
		
		int pgas_tile_size() const { return get_tile_size()*get_tile_size()*sizeof(T); }
		
		void clear_cache() {
			for (int i = 0; i<get_nb_tile_y()*get_nb_tile_x(); ++i) {
				if (_data[i] != 0)
					__sync_lock_test_and_set (&(_data[i]->flag), 0);
			}
		}
};
		


template <typename T, int tile_size>
T* remote_matrix<T, tile_size>::get_tile_unitialized(const int x, const int y) {
	#pragma omp critical
	{	
		if (_data[y*_nb_tiles_x + x] == 0) {
			_data[y*_nb_tiles_x + x] = new tile();
		}
	}
	 
	return _data[y*_nb_tiles_x + x]->data;
}

template <typename T, int tile_size>
void remote_matrix<T, tile_size>::set_tile(T const * restrict const ptr, const int x, const int y, const bool sent) {
	GASNET_BEGIN_FUNCTION();
	
	using namespace adabs::tools;

	if (_data[y*_nb_tiles_x + x] == 0 || _data[y*_nb_tiles_x + x]->data != ptr) {
		throw "Error";
	}
	
	__sync_lock_test_and_set (&_data[y*_nb_tiles_x + x]->flag, 2);
	
	
	if (sent) {
		// dataT is tile*
		tile *dest_ptr1 = static_cast<tile*>(pgas_get_data_ptr());
		      dest_ptr1 += y*_nb_tiles_x + x;
		void *dest_ptr2 = (void*)(dest_ptr1);
		
		
		GASNET_CALL(gasnet_AMRequestLong4(_addr.get_node(),
					                       adabs::impl::MATRIX_BASE_SET,
					                       (void*)_data[y*_nb_tiles_x + x]->data, pgas_tile_size(),
					                       dest_ptr2,
					                       get_low(_addr.get_ptr()),
					                       get_high(_addr.get_ptr()),
					                       x, y
					                      )
				   )
	}
}

template <typename T, int tile_size>
T const* remote_matrix<T, tile_size>::get_tile(const int x, const int y) {
	GASNET_BEGIN_FUNCTION();
	
	using namespace adabs::tools;
	
	// in cache?
	if (_data[y*_nb_tiles_x + x]!= 0 && _data[y*_nb_tiles_x + x]->flag == 2) {
		//std::cout << gasnet_mynode() << " cached" << std::endl;
		return _data[y*_nb_tiles_x + x]->data;
	}

	// get dest_ptr here to make sure the tile exists
	T* dest_ptr = get_tile_unitialized(x, y);
	bool request = __sync_bool_compare_and_swap (&_data[y*_nb_tiles_x + x]->flag, 0, 1);	
	
	// get data from source
	//std::cout << gasnet_mynode() << " remote" << std::endl;
	if (request) {
		//std::cout << gasnet_mynode() << " request startet for " << x << ", " << y << std::endl;
		GASNET_CALL(gasnet_AMRequestShort8(_addr.get_node(),
						                   adabs::impl::MATRIX_BASE_GET,
						                   get_low(_addr.get_ptr()),
						                   get_high(_addr.get_ptr()),
						                   get_low(this),
						                   get_high(this),
						                   get_low(dest_ptr),
						                   get_high(dest_ptr),
						                   x, y
						                  )
				   )
	}

	//volatile int *reader = &_data[y*_nb_tiles_x + x]->flag; 
	//while (*reader != 2){}
	GASNET_BLOCKUNTIL(_data[y*_nb_tiles_x + x]->flag == 2);
	
	return _data[y*_nb_tiles_x + x]->data;
}

}
