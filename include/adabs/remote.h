#pragma once

#include "adabs/adabs.h"
#include "adabs/pgas_addr.h"
#include "adabs/memcpy.h"
#include "adabs/tools/ptr_divider.h"

namespace adabs {

template <typename T>
class local;

/**
 * Maybe BYTEWISE COPIED after the copy constructor!
 * Make sure the copy will work fine even on different nodes
 */
template <typename T>
class remote {
	public:
		typedef T value_type;
		
	private:
		int _x;
		int _y;
		int _batch_size_x;
		int _batch_size_y;
		
		pgas_addr<T> _data;
		mutable pgas_addr<T>** _datas;
		
	public:
		template <typename cpy_distT>
		remote(const cpy_distT& cpy_dist) :
		   _x(cpy_dist.get_size_x()), _y(cpy_dist.get_size_y()),
		   _batch_size_x(cpy_dist.get_batch_size_x()), _batch_size_y(cpy_dist.get_batch_size_y()),
		   _data(cpy_dist.get_data_addr()),
		   _datas(0)
		 {
			assert (_x%_batch_size_x == 0);
			assert (_y%_batch_size_y == 0);
			
		}
		

		remote(const remote<T> &cpy) :
		   _x(cpy.get_size_x()), _y(cpy.get_size_y()),
		   _batch_size_x(cpy.get_batch_size_x()), _batch_size_y(cpy.get_batch_size_y()),
		   _data(cpy._data),
		   _datas(0)
		{ }
		
		~remote() {
			if (_datas != 0) {
				for (int i=0; i<(_x/_batch_size_x) * (_y/_batch_size_y); ++i)
					delete _datas[i];
			}
			delete[] _datas;
		}
		
	private:
	
	public:
		const pgas_addr<T>& get_data_addr() const {
			return _data;
		}
		T* get_data_unitialized(const int x, const int y=1) {
			assert (x%_batch_size_x == 0);
			assert (y%_batch_size_y == 0);
			
			const int offset = get_offset(x,y);
			
			//std::cout << me << ": get uninit " << x << " - " << y << " - " << offset << std::endl;
			
			#pragma omp critical (remote_cache)
			{
				if (_datas == 0) init();
				if (_datas[offset] == 0)
					_datas[offset] = new pgas_addr<T>((*_datas[0]) + offset);
			}
			
			return _datas[offset]->get_data_unitialized();
		}
		
		const T* get_data(const int x, const int y=1) const {
			assert (x%_batch_size_x == 0);
			assert (y%_batch_size_y == 0);
			
			//std::cout << me << ": " << "remote get data called" << std::endl;
			
			const int offset = get_offset(x,y);
			
			#pragma omp critical (remote_cache)
			{
				if (_datas == 0) init();
				if (_datas[offset] == 0)
					_datas[offset] = new pgas_addr<T>((*_datas[0]) + offset);
			}
			
			return _datas[offset]->get_data();
		}

		void set_data(const int x, T const * restrict const ptr) {
			assert (x%_batch_size_x == 0);
			assert (_datas != 0);
			
			const int offset = get_offset(x,1);
			
			_datas[offset]->set_data(ptr);
		}		
		
		void set_data(const int x, const int y, T* ptr) {
			assert (x%_batch_size_x == 0);
			assert (y%_batch_size_y == 0);
			assert (_datas != 0);
			
			const int offset = get_offset(x,y);
			
			_datas[offset]->set_data(ptr);
		}

		void fill(const int x, const int y, const pgas_addr<T> ptr, const int nb_elements) {
			assert (x%_batch_size_x == 0);
			assert (y%_batch_size_y == 0);
			assert(_batch_size_x * _batch_size_y == ptr.get_batch_size());
			assert((_batch_size_x * _batch_size_y)%nb_elements == 0);
			
			// TODO do not use!!
			
			const int batches = nb_elements / _batch_size_x / _batch_size_y;
			const int offset  = get_offset(x,y);
			
			adabs::memcpy(_data+offset, ptr, batches);
			
			// TODO fill cache?
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
		
		remote<T>& operator=(const local<T> &rhs) {
			using namespace adabs::tools;
			
			for (int i=0; i<_y; i+=_batch_size_y) {
				for (int j=0; j<_x; j+=_batch_size_x) {
					rhs.get_data(i, j);
				}
			}
			
			// TODO add options to disable checks
			{
				pgas_addr<T> temp = rhs.get_data() + 1;
				const int stride = (char*)temp.get_flag() - (char*)rhs.get_data().get_flag();
			
				// check if remote data is still empty
				volatile int done = 0;
				GASNET_CALL(gasnet_AMRequestShort6(_data.get_node(),
											       adabs::impl::PGAS_ADDR_GET_UNINIT,
											       get_low(_data.get_orig_flag()),
											       get_high(_data.get_orig_flag()),
											       stride,
											       local_size()/_batch_size_x/_batch_size_y,
											       get_low(&done),
											       get_high(&done)
											      )
					      )

				while (done != 1) {}
			}
			
			adabs::memcpy(_data, rhs.get_data(), local_size()/_batch_size_x/_batch_size_y);
			
			return *this;
		}

	private:
		void init() const {
			assert(_datas == 0);
			
			_datas = new pgas_addr<T>*[(_x/_batch_size_x) * (_y/_batch_size_y)];
			_datas[0] = new pgas_addr<T>(_data);
			for (int i=1; i<(_x/_batch_size_x) * (_y/_batch_size_y); ++i)
				_datas[i] = 0;
			
			assert (_datas != 0);
		}
		
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
