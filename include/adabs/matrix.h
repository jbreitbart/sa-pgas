#pragma once

#include "adabs/adabs.h"
#include "adabs/remote.h"

namespace adabs {

/**
 * This is a tile based matrix class.
 *
 * Note: You should not inheriate from this matrix.
 */
template <typename T>
class matrix {
		/************************ TYPEDEFS ***************************/
	private:

	public:
		typedef typename T::value_type value_type;
		
	
		/************************ VARIABLES **************************/
	private:
		T _distri;

		
		/********************* CON/DESTRUCTOS ************************/
	private:
		
	public:
		matrix (const T& distri_cpy) : _distri(distri_cpy)
		{}
		
		/**
		 * Creates a matrix of size
		 * @param size_x * @param size_y
		 * . In case the size is not a multiply of the tile_size
		 * the size of the matrix is increased so it is a multiple of
		 * tile_size
		 */
		matrix (const int size_x, const int size_y, const int batch_size)
		  : _distri(size_x, size_y, batch_size, batch_size)
		{}
		
		/**
		 * Copy constructor to create a copy of @param cpy
		 */
		matrix (const matrix<T>& cpy) : _distri (cpy.get_distri().get_x(), cpy.get_distri().get_y(), cpy.get_distri().get_batch_size()) {
			_distri = cpy.get_distri();
		}
		
		
		/**
		 * Desctructor, make sure to not delete the object before(!) all
		 * reads to that matrix are completed.
		 */
		~matrix() { }
		
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
		void set_tile(const int x, const int y, typename T::value_type * ptr);
		
		/**
		 * Returns a pointer to the tile with the coordinated
		 * @param x and @param y
		 * . In case the values are not yet written to the matrix, the
		 * calling thread will sleep until the value is returned.
		 */
		typename T::value_type const* get_tile(const int x, const int y) const;
		
		/**
		 * Returns the pointer to the matrix internal tile with the
		 * coordinates
		 * @param x and  @param y
		 * so one can update the matrix in place. You must(!) still call
		 * set_tile() for this matrix tile!
		 */
		typename T::value_type* get_tile_unitialized(const int x, const int y);
		
		/**
		 * Returns the tile size
		 */
		int get_tile_size() const { return _distri.get_batch_size_x(); }
		
		/**
		 * Returns a pgas address for the current matrix
		 */
		matrix< remote <typename T::value_type> >make_remote() const {
			return matrix<remote <typename T::value_type> >(_distri.make_remote());
		}

		bool is_local(const int x, const int y) {
			return _distri.is_local(x,y);
		}		
};

template <typename T>
typename T::value_type* matrix<T>::get_tile_unitialized(const int x, const int y) {
	return _distri.get_data_unitialized(x,y);
}

template <typename T>
void matrix<T>::set_tile(const int x, const int y, typename T::value_type * ptr) {
	_distri.set_data(x, y, ptr);
}

template <typename T>
typename T::value_type const* matrix<T>::get_tile(const int x, const int y) const {
	return _distri.get_data(x,y);
}

}
