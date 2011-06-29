#pragma once

#include "adabs/gasnet_config.h"

#include "adabs/matrix_base.h"

#include "pgas_addr.h"
#include "tools/tools.h"
#include "tools/tile.h"

namespace adabs {

/**
 * This is a tile based matrix class.
 *
 * Note: You should not inheriate from this matrix.
 */
template <typename T, int tile_size>
class matrix : public matrix_base {
		/************************ TYPEDEFS ***************************/
	private:
		typedef tools::tile<T, tile_size> tile;
		typedef std::pair<volatile int, tile> dataT;
		
	
		/************************ VARIABLES **************************/
	private:
		dataT* _data;
		const int _nb_tiles_x;
		const int _nb_tiles_y;
				
		/********************* CON/DESTRUCTOS ************************/
	private:
		/**
		 * This is a constructor helper, allocating the memory required
		 * for the matrix. Requires _size_{x|y} to be set.
		 */
		void alloc_memory();
		
	public:
		/**
		 * Creates a matrix of size
		 * @param size_x * @param size_y
		 * . In case the size is not a multiply of the tile_size
		 * the size of the matrix is increased so it is a multiple of
		 * tile_size
		 */
		matrix (const int size_x, const int size_y)
		  : matrix_base(size_x, size_y),
		    _nb_tiles_x((get_size_x()%tile_size == 0) ? (get_size_x()/tile_size) : (get_size_x()/tile_size+1)),
		    _nb_tiles_y((get_size_y()%tile_size == 0) ? (get_size_y()/tile_size) : (get_size_y()/tile_size+1)) {
			alloc_memory();
		}
		
		/**
		 * Copy constructor to create a copy of @param cpy
		 */
		matrix (const matrix<T, tile_size>& cpy) : matrix_base(cpy), _nb_tiles_x(cpy._nb_tiles_x), _nb_tiles_y(cpy._nb_tiles_y) {
			alloc_memory();
			
			const int tiles_x = get_nb_tile_x();
			const int tiles_y = get_nb_tile_y();
			
			for (int y=0; y<tiles_y; ++y) {
				for (int x=0; x<tiles_x; ++x) {
					if (cpy._data[y*_nb_tiles_x + x].first == 1) {
						copy_tile(cpy._data[y*_nb_tiles_x + x].second.data, x, y);
					}
				}
			}
		}
		
		/**
		 * Desctructor, make sure to not delete the object before(!) all
		 * reads to that matrix are completed.
		 */
		~matrix() { delete[] _data; }
		
		/************************ FUNCTIONS **************************/
	private:
		void copy_tile(const restrict T *const source, const int x, const int y) {
			restrict T *const target = _data[y*_nb_tiles_x + x].second.data;
			
			#pragma vector aligned nontemporal(target, source)
			for (int i=0; i<tile_size*tile_size; ++i)
				target[i] = source[i];
			
			__sync_lock_test_and_set (&_data[y*_nb_tiles_x + x].first, 1);
		}


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
		void set_tile(T const * restrict const ptr, const int x, const int y);
		
		/**
		 * Returns a pointer to the tile with the coordinated
		 * @param x and @param y
		 * . In case the values are not yet written to the matrix, the
		 * calling thread will sleep until the value is returned.
		 */
		T const* get_tile(const int x, const int y) const;
		
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
		
		/**
		 * Returns a pgas address for the current matrix
		 */
		pgas_addr <matrix<T, tile_size> > get_pgas_addr() {
			return pgas_addr < matrix<T, tile_size> > (gasnet_mynode(), this);
		}

		void* pgas_get(const int x, const int y) const {
			return (void*) get_tile(x, y);
		}
		
		void pgas_mark(const int x, const int y) {
			//std::cout << gasnet_mynode() << " " << "pgas mark " << y*_nb_tiles_x + x << " - " 
			// << " " << &_data[y*_nb_tiles_x + x].second.data[0] << std::endl;
			__sync_lock_test_and_set (&_data[y*_nb_tiles_x + x].first, 1);
			//std::cout << gasnet_mynode() << " new value: " << _data[y*_nb_tiles_x + x].first << std::endl;
		}
		
		int pgas_tile_size() const { return get_tile_size()*get_tile_size()*sizeof(T); }
		
		void reuse() {
			const int tiles_x = get_nb_tile_x();
			const int tiles_y = get_nb_tile_y();
			
			for (int y=0; y<tiles_y * tiles_x; ++y) {
				__sync_lock_test_and_set (&_data[y].first, 0);
			}

		}

		bool is_local(const int x, const int y) {
			return true;
		}		
};

template <typename T, int tile_size>
void matrix<T, tile_size>::alloc_memory() {
	const int tiles_x = get_nb_tile_x();
	const int tiles_y = get_nb_tile_y();
	
	_data = new dataT[tiles_y*tiles_x];
	pgas_set_data_ptr((void*)_data);
}

template <typename T, int tile_size>
T* matrix<T, tile_size>::get_tile_unitialized(const int x, const int y) {
	return _data[y*_nb_tiles_x + x].second.data;
}

template <typename T, int tile_size>
void matrix<T, tile_size>::set_tile(T const * restrict const ptr, const int x, const int y) {
	if (_data[y*_nb_tiles_x + x].second.data != ptr) {
		throw "Error";
	}

	__sync_lock_test_and_set (&_data[y*_nb_tiles_x + x].first, 1);
	__sync_synchronize();
}

template <typename T, int tile_size>
T const* matrix<T, tile_size>::get_tile(const int x, const int y) const {
	//std::cout << gasnet_mynode() << " " << "local get_tile " << x << " " << y << " - " << &_data[y*_nb_tiles_x + x].first << std::endl;
	//std::cout << gasnet_mynode() << " " << _data[y*_nb_tiles_x + x].first << " - " << &_data[y*_nb_tiles_x + x].first << std::endl;
	
	volatile int *reader = &_data[y*_nb_tiles_x + x].first;
	while (*reader == 0){}
	//GASNET_BLOCKUNTIL(_data[y*_nb_tiles_x + x].first != 0);
	
	return _data[y*_nb_tiles_x + x].second.data;
}

}
