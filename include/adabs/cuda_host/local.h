#pragma once

#include "adabs/adabs.h"
#include "adabs/remote.h"
#include "adabs/cuda_host/pgas_addr.h"

namespace adabs {

namespace cuda_host {

template <typename T>
class local {
	public:
		typedef T value_type;
		
	private:
		const int _x;
		const int _y;
		const int _batch_size_x;
		const int _batch_size_y;
		
		pgas_addr<T> _data;
		
	public:
		local(const int x, const int y, const int batch_size_x, const int batch_size_y) :
		    _x(x), _y(y), _batch_size_x(batch_size_x), _batch_size_y(batch_size_y),
		    _data (adabs::cuda_host::allocator<T>::allocate(_x*_y, _batch_size_x*_batch_size_y))
		 {
			assert (_x%_batch_size_x == 0);
			assert (_y%_batch_size_y == 0);
		}
		
		local(const int x, const int batch_size_x) : _x(x), _y(1), _batch_size_x(batch_size_x), _batch_size_y(1),
		    _data (adabs::allocator<T>::allocate(_x*_y, _batch_size_x*_batch_size_y))
		 {
			assert (_x%_batch_size_x == 0);
			assert (_y%_batch_size_y == 0);
		}
		
		~local() {
			adabs::cuda_host::allocator<T>::deallocate(_data);
		}
	
	private:
		local(const local<T> &cpy);// : _x(cpy._x), _y(cpy._y), _batch_size_x(cpy._batch_size_x), _batch_size_y(cpy._batch_size_y), _data(cpy._data)
		//{}
	
	public:
		remote<T> make_remote() const {
			return remote<T> (*this);
		}
		
		pgas_addr<T> get_data_addr() const {
			return _data;
		}
		
		T* get_data_unitialized(const int x, const int y=1) {
			assert (x%_batch_size_x == 0);
			assert (y%_batch_size_y == 0);
			
			const int offset = get_offset(x,y);
			
			return (_data + offset).get_data_unitialized();
		}
		
		T* get_data(const int x, const int y=1) const {
			assert (x%_batch_size_x == 0);
			assert (y%_batch_size_y == 0);
			
			const int offset = get_offset(x,y);
			
			return (_data + offset).get_data();
		}

		void set_data(const int x, T const * restrict const ptr) {
			assert (x%_batch_size_x == 0);
			assert (_y == 1);
			
			const int offset = get_offset(x,1);
			
			(_data + offset).set_data(ptr);
		}		
		
		void set_data(const int x, const int y, T* ptr) {
			assert (x%_batch_size_x == 0);
			assert (y%_batch_size_y == 0);
			
			const int offset = get_offset(x,y);
			
			(_data + offset).set_data(ptr);
		}
		
		int get_size_x() const {
			return _x;
		}
		
		int get_size_y() const {
			return _y;
		}
		
		int get_batch_size_x() const {
			return _batch_size_x;
		}
		
		int get_batch_size_y() const {
			return _batch_size_y;
		}
		
		pgas_addr<T>& get_data() const {
			return const_cast<pgas_addr<T>&>(_data);
		}
		
	private:
		int local_size() const {
			return _x*_y;
		}
		
		bool is_local(const int x, const int y) const {
			return get_node(x,y) == adabs::me;
		}
		
		int get_node(const int x, const int y) const {
			return _data.get_node();
		}
		
		int get_local_x(const int x) const {
			return x;
		}
		
		int get_local_y(const int y) const {
			return y;
		}
		
		int get_offset(const int x, const int y) const {
			if (_y==1)
				return x / _batch_size_x;
			else
				return (x/_batch_size_x + (_x/_batch_size_x)*(y/_batch_size_y));
		}
};


}
}
