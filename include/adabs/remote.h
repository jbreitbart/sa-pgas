#pragma once

#include "adabs/adabs.h"
#include "adabs/pgas_addr.h"

namespace adabs {

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
			assert (y == 1);
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
