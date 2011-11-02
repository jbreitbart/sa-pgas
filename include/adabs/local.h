#pragma once

#include "adabs/adabs.h"
#include "adabs/remote.h"
#include "adabs/pgas_addr.h"
#include "adabs/memcpy.h"
#include "adabs/distributed/row_distribution.h"

namespace adabs {

namespace distributed {
template <typename T, int nb_of_rows> class row_distribution;
}

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
		    _data (adabs::allocator<T>::allocate(_x*_y, _batch_size_x*_batch_size_y))
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
			adabs::allocator<T>::deallocate(_data);
		}
	
	private:
		local(const local<T> &cpy);
	
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
		
		local<T>& operator=(const local<T> &rhs) {
			assert (_x == rhs._x);
			assert (_y == rhs._y);
			assert (_batch_size_x == rhs._batch_size_x);
			assert (_batch_size_y == rhs._batch_size_y);
			
			for (int i=0; i<_y; i+=_batch_size_y) {
				for (int j=0; j<_x; j+=_batch_size_x) {
					rhs.get_data(i, j);
					get_data_unitialized(i, j);
				}
			}
			
			adabs::memcpy(_data, rhs._data, local_size()/_batch_size_x/_batch_size_y);
			
			return *this;
		}
		
		void wait_for_complete() const {
			for (int i=0; i<_y; i+=_batch_size_y) {
				for (int j=0; j<_x; j+=_batch_size_x) {
					get_data(j, i);
				}
			}
		}
		
		void check_empty() const; // TODO

		local<T>& operator=(const remote<T> &rhs) {
			using namespace adabs::tools;
			assert (_x == rhs.get_size_x());
			assert (_y == rhs.get_size_y());
			assert (_batch_size_x == rhs.get_batch_size_x());
			assert (_batch_size_y == rhs.get_batch_size_y());

			for (int i=0; i<_y; i+=_batch_size_y) {
				for (int j=0; j<_x; j+=_batch_size_x) {
					get_data_unitialized(i, j);
				}
			}
			
			rhs.wait_for_complete();
			
			adabs::memcpy(_data, rhs.get_data_addr(), local_size()/_batch_size_x/_batch_size_y);
			
			return *this;
		}
		
		template <int N>
		local<T>& operator=(const distributed::row_distribution<T, N> &rhs) {
			using namespace adabs::tools;
			assert (_x == rhs.get_size_x());
			assert (_y == rhs.get_size_y());
			assert (_batch_size_x == rhs.get_batch_size_x());
			assert (_batch_size_y == rhs.get_batch_size_y());

			rhs.wait_for_complete();
			
			local<T> temp (_x, _y, _batch_size_x, _batch_size_y);
			
			for (int i=0; i<_y; i+=_batch_size_y) {
				for (int j=0; j<_x; j+=_batch_size_x) {
					temp.get_data_unitialized(i, j);
				}
			}
			
			//fill temp with data
			pgas_addr<T> temp_data = temp.get_data();
			for (int i=0; i<adabs::all; ++i) {
				const int batches = rhs.get_distri(i).get_size_x()*rhs.get_distri(i).get_size_y() / _batch_size_x / _batch_size_y;
				adabs::memcpy(temp_data, rhs.get_distri(i).get_data_addr(), batches);
				temp_data += batches;
			}
			
			//copy data in this
			for (int i=0; i<_y; i+=_batch_size_x) {
				for (int j=0; j<_x; j+=_batch_size_y) {
					const int s_x = rhs.get_local_x(j);
					const int s_y = rhs.get_local_y(i);
					const int s_h = rhs.get_node(j, i);
					
					int global_offset = 0;
					for (int k=0; k<s_h; ++k) global_offset += rhs.local_size_y(k);
					
					restrict T* ptr = get_data_unitialized(i, j);
					const restrict T* temp_ptr = temp.get_data(s_x, global_offset + s_y);
					
					for (int k=0; k<_batch_size_x*_batch_size_y; ++k) {
						ptr[k] = temp_ptr[k];
					}
					
					set_data(i, j, ptr);
					
				}
			}

			return *this;
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
