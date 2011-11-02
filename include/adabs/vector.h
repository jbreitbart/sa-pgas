#pragma once

#include "adabs/collective/allocator.h"
#include "adabs/collective/pgas_addr.h"

namespace adabs {

template <typename T>
class vector {
	
		/************************ VARIABLES **************************/
	private:
		T _distri;
				
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
		vector (const int size, const int batch_size = 1) :
		   _distri(size, batch_size)
		{}
		
		/**
		 * Copy constructor to create a copy of @param cpy
		 */
		vector (const vector<T>& cpy) :
		   _distri(cpy.size(), cpy.get_batch_size())
		{
			_distri = cpy._distri;
		}
		
		/**
		 * Desctructor, make sure to not delete the object before(!) all
		 * reads to that matrix are completed.
		 */
		~vector() {}
		
		/************************ FUNCTIONS **************************/
	private:

	public:
		 void set(const int x, typename T::value_type * ptr) {
		 	_distri.set_data(x, ptr);
		}

		const typename T::value_type& get(const int x) const {
			// TODO fix this, so it will work with non % batch_size == 0
			return *_distri.get_data(x);
		}
		
		const typename T::value_type* get_tile(const int x) const {
			return _distri.get_data(x);
		}
		
		typename T::value_type* get_unitialized(const int x) {
			return _distri.get_data_unitialized(x);
		}
		
		/**
		 * Returns the number of tiles in x-dimension
		 */
		int size() const {
			return _distri.get_size_x();
		}
		
		int get_batch_size() const {
			return _distri.get_batch_size();
		}

		template <typename T2>
		vector<T>& operator=(const vector<T2>& rhs) {
			return _distri = rhs._distri;
		}
		
		const T& get_distri() const {
			return _distri;
		}
};

}
