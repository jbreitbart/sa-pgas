#pragma once

#include "adabs/adabs.h"
#include "adabs/remote.h"
#include "adabs/pgas_addr.h"
#include "adabs/memcpy.h"

namespace adabs {

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
		
		void fill(const int x, const int y, pgas_addr<T> ptr, const int nb_elements) {
			assert (x%_batch_size_x == 0);
			assert (y%_batch_size_y == 0);
			assert(_batch_size_x * _batch_size_y == ptr.get_batch_size());
			assert((_batch_size_x * _batch_size_y)%nb_of_elements == 0);
			
			// TODO check for data alread there
			
			const int batches = nb_elements / _batch_size_x / _batch_size_y;
			const int offset  = get_offset(x,y);
			
			adabs::memcpy(_data+offset, ptr, batches);
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
			
			{
				pgas_addr<T> temp = rhs.get_data_addr() + 1;
				const int stride = (char*)temp.get_orig_flag() - (char*)rhs.get_data_addr().get_orig_flag();
			
				// check if remote data is available
				volatile int done = 0;
				
				GASNET_CALL(gasnet_AMRequestShort6(rhs.get_data_addr().get_node(),
											       adabs::impl::PGAS_ADDR_CHECK_GET_ALL,
											       get_low(rhs.get_data_addr().get_orig_flag()),
											       get_high(rhs.get_data_addr().get_orig_flag()),
											       stride,
											       local_size()/_batch_size_x/_batch_size_y,
											       get_low(&done),
											       get_high(&done)
											      )
					      )

				while (done != 1) {}
			}
			
			adabs::memcpy(_data, rhs.get_data_addr(), local_size()/_batch_size_x/_batch_size_y);
			
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
