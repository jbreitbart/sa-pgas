#pragma once

#include <vector>

#include "adabs/collective/vector_base.h"

namespace adabs {

namespace collective {

template <typename T>
class vector : public vector_base {
		/************************ TYPEDEFS ***************************/
	private:
		typedef std::pair< int, T > dataT;
		
	
		/************************ VARIABLES **************************/
	private:
		std::vector <dataT> _data;
				
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
		vector (const int size) : vector_base(), _data(size) {
				barrier_wait();
		}
		
		/**
		 * Copy constructor to create a copy of @param cpy
		 */
		vector (const vector<T>& cpy) : vector_base(cpy), _data(cpy._data) {
		}
		
		/**
		 * Desctructor, make sure to not delete the object before(!) all
		 * reads to that matrix are completed.
		 */
		~vector() {}
		
		/************************ FUNCTIONS **************************/
	private:

	public:
		 void set(const int x, const void* const ptr) {
			set(x, *(T*)ptr, false);
		}

		void set (const int x, const T& val, bool broadcast = true) {
			using namespace adabs::tools;
			using adabs::me;
			using adabs::all;
			
			// will we broadcast the values to all processes
			if (broadcast) {
				for (int i=0; i<all; ++i) {
						if (i==me) continue;
						GASNET_CALL(gasnet_AMRequestMedium3(i,
									                       adabs::impl::COLLECTIVE_VECTOR_SET,
									                       (void*)&val, sizeof(T),
									                       get_low(_proxies[i]),
									                       get_high(_proxies[i]),
									                       x
									                      )
								   )
				}
			}
			
			_data[x].second = val;
			__sync_lock_test_and_set (&_data[x].first, 1);
		}
		
		const T& get(const int x) const {
			GASNET_BLOCKUNTIL(_data[x].first==1);
			
			return _data[x].second;
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
