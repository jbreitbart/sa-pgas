#pragma once

#include "adabs/collective/allocator.h"
#include "adabs/collective/pgas_addr.h"

namespace adabs {

namespace collective {

template <typename T>
class vector {
	
		/************************ VARIABLES **************************/
	private:
		pgas_addr<T> _data;
		const int _size;
		const int _batch_size;
				
		/********************* CON/DESTRUCTOS ************************/
	private:
		
	public:
		/**
		 * Creates a matrix of size
		 * @param size_x * @param size_y
		 * . In case the size is not a multiply of the tile_size
		 * the size of the matrix is increased so it is a multiple of
		 * tile_size
		 */
		vector (const int size, const int batch_size = 1) : _data(allocator<T>::allocate(size, batch_size)), _size(size), _batch_size(batch_size) {
		}
		
		/**
		 * Copy constructor to create a copy of @param cpy
		 */
		vector (const vector<T>& cpy) : _data(allocator<T>::allocate(size, batch_size)), _size(cpy._size), _batch_size(cpy._batch_size) {
			for (int i=0; i<size; ++i) {
				set(i, cpy.get(i));
			}
		}
		
		/**
		 * Desctructor, make sure to not delete the object before(!) all
		 * reads to that matrix are completed.
		 */
		~vector() {
			if (leader)
				allocator<T>::deallocate(_data);
		}
		
		/************************ FUNCTIONS **************************/
	private:

	public:
		 void set(const int x, const T* const ptr) {
		 	assert(x % _batch_size == 0);
		 	(_data+x).set_data(ptr);
		}

		const T& get(const int x) const {
			// TODO fix this, so it will work with non % batch_size == 0
			assert(x % _batch_size == 0);
			return *(_data+x).get_data();
		}
		
		T* get_unitialized(const int x) {
			assert(x % _batch_size == 0);
			return (_data+x).get_data_unitialized();
		}
		
		/**
		 * Returns the number of tiles in x-dimension
		 */
		int size() const {
			return _data.size();
		}

};

}

}
