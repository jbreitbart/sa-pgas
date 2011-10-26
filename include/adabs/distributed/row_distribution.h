#pragma once

#include "adabs/adabs.h"
#include "adabs/vector.h"
#include "adabs/collective/everywhere.h"
#include "adabs/local.h"

namespace adabs {
namespace distributed {

template <typename T, int nb_of_rows>
class row_distribution {
	public:
		typedef T value_type;
		
	private:
		const int _x;
		const int _y;
		const int _batch_size_x;
		const int _batch_size_y;
		
		adabs::vector< adabs::collective::everywhere< adabs::remote<T> > > _remote;
		adabs::local<T> _local;
		
	public:
		row_distribution(const int x, const int y, const int batch_size_x, const int batch_size_y) :
		    _x(x), _y(y), _batch_size_x(batch_size_x), _batch_size_y(batch_size_y),
			_remote(adabs::all),
			_local(x, local_size_y(), _batch_size_x, _batch_size_y)
		 {
			assert (_x%_batch_size_x == 0);
			assert (_y%_batch_size_y == 0);
			assert (nb_of_rows % _batch_size_y == 0);
			
			adabs::remote<T>* ptr = _remote.get_unitialized(adabs::me);
			new (ptr) adabs::remote<T>(_local);
			_remote.set(adabs::me, ptr);
		}
		
		~row_distribution() {
		}
	
	private:
		row_distribution(const row_distribution<T, nb_of_rows> &copy);
	
	public:
		T* get_data_unitialized(const int x, const int y) {
			assert (x%_batch_size_x == 0);
			assert (y%_batch_size_y == 0);
			
			if (is_local(x,y)) {
				return _local.get_data_unitialized(get_local_x(x), get_local_y(y));
			}
			
			
			const int node = get_node(x,y);
			
			return const_cast< adabs::remote<T>& >(_remote.get(node)).get_data_unitialized(get_local_x(x), get_local_y(y));
		}
		
		const T* get_data(const int x, const int y) const {
			assert (x%_batch_size_x == 0);
			assert (y%_batch_size_y == 0);
			
			if (is_local(x,y)) {
				//std::cout << "local get" << std::endl;
				return _local.get_data(get_local_x(x), get_local_y(y));
			}
			
			
			const int node = get_node(x,y);
			//std::cout << "remote get from node " << node << std::endl;
			const adabs::remote<T> &temp = _remote.get(node);
			//std::cout << "address " << get_local_x(x) << ", " << get_local_y(y) << std::endl;
			return temp.get_data(get_local_x(x), get_local_y(y));
		}
		
		void set_data(const int x, const int y, T* ptr) {
			assert (x%_batch_size_x == 0);
			assert (y%_batch_size_y == 0);
			
			//std::cout << "set dist data " << x << ", " << y << " local on " << get_node(x,y) << " at " << get_local_x(x) << ", " << get_local_y(y) << std::endl;
			if (is_local(x,y)) {
				_local.set_data(get_local_x(x), get_local_y(y), ptr);
				return;
			}
			
			
			const int node = get_node(x,y);
			
			const_cast< adabs::remote<T>& >(_remote.get(node)).set_data(get_local_x(x), get_local_y(y), ptr);
		}
		
		/*void fill(const int x, const int y, pgas_addr<T> ptr, const int nb_elements) {
			assert (x%_batch_size_x == 0);
			assert (y%_batch_size_y == 0);
			assert(_batch_size_x * _batch_size_y == ptr.get_batch_size());
			assert((_batch_size_x * _batch_size_y)%nb_of_elements == 0);
			assert(is_local(x,y));
			// the row of x is local for our distribution

			const int batches = nb_elements / _batch_size_x / _batch_size_y;
			
			{
				int temp = batches - (_local.get_size_x()-x)/_batch_size_x;
				for (int j=y+1; temp>0; temp-=_local.get_size_x()/_batch_size_x, j+=_batch_size_y) {
					assert (is_local(x, j));
				}
			}
			
			const int offset  = get_offset(x,y);
			
			adabs::memcpy(_local+offset, ptr, batches);
		}*/
		
		int get_batch_size_x() const {
			return _batch_size_x;
		}
		
		int get_batch_size_y() const {
			return _batch_size_y;
		}
		
	private:
		
		int local_size_y() const {
			const int y_t = (_y%adabs::all == 0) ? _y/adabs::all : _y/adabs::all+1;
			
			return y_t;
		}
		
		bool is_local(const int x, const int y) const {
			return get_node(x,y) == adabs::me;
		}
		
		int get_node(const int x, const int y) const {
			return (y/nb_of_rows) % adabs::all;
		}
		
		int get_local_x(const int x) const {
			return x;
		}
		
		int get_local_y(const int y) const {
			return y/(all*nb_of_rows)*nb_of_rows + (y%nb_of_rows);
		}
		
		int get_offset(const int x, const int y) const {
			return (get_local_x(x)/_batch_size_x + _x*get_local_y(y)/_batch_size_x/_batch_size_y);
		}
};



}
}

#if 0
/**
 * Simple row distribution used with distributed matrixes.
 * Mostly a very basic example for distributions, nothing fancy.
 * Interface may need some redesign, but we need more examples to be
 * sure.
 * @param row_size defines the size of the rows
 */
template <int row_size>
struct row_distribution {
	static int get_local_x(const int x) {
		return x;
	}
	
	static int get_local_y(const int y) {
		using adabs::all;
		
		//const int tile = y / tile_size;
		const int result = y/row_size/all*row_size + (y%row_size);
		//std::cout << y << "::::: " << result << ", " << y/row_size/all*row_size << ", " << (y%row_size) << std::endl;
		
		//const int returnee = result*tile_size + (y%tile_size);
		
		return result;
	}
	
	static int get_node(const int x, const int y) {
		using adabs::all;

		
		return (y/row_size)%all;
	}
	
	static bool is_local(const int x, const int y) {
		using adabs::me;
		
		return get_node(x,y) == me;
	}
	
	static int get_local_size_x(const int size_x) {
		return size_x;
	}
	
	template <int tile_size>
	static int get_local_size_y(const int size_y) {
		using adabs::all;
		const int nb_tiles = (size_y%tile_size == 0) ? (size_y/tile_size) : (size_y/tile_size+1);
		
		const int a = (nb_tiles%row_size) == 0 ? nb_tiles / row_size : nb_tiles / row_size+1;
		const int b = (a%all) == 0 ? a/all*row_size : (a/all+1)*row_size;
		
		if (b<row_size) return tile_size*row_size;
		else return tile_size*b;
	}
	
	template <int tile_size, typename T, typename T2>
	static void scatter_root(T const &rhs, const T2 &lhs) {
		
		typedef typename T::value_type valT;
		
		int *starts = new int[all];
		
		struct transferT {
			adabs::pgas_addr<valT> next_remote;
			T2* local_dist;
			valT* data;
		};
		
		// allocate remote memory
		adabs::pgas_addr<valT>* remote_addrs = new adabs::pgas_addr<valT>[adabs::all]; 
		
		//TODO this relies on the current implementation of next being (X+1)%all
		int mem = get_local_size_y<tile_size>(rhs.get_nb_tile_y()*tile_size) * (all-1);
		for (int i=adabs::next; i!=adabs::me; i=(i+1)%all) {
			remote_addrs[i] = adabs::remote_malloc<valT>(i, mem);
			mem -= get_local_size_y<tile_size>(rhs.get_nb_tile_y()*tile_size);
		}
		{
			int start=0;
			for (int i=adabs::next; i!=adabs::me; i=(i+1)%all) {
				starts[i] = start;
				std::cout << i << " - " << start << std::endl;
				start += get_local_size_y<tile_size>(rhs.get_nb_tile_y()*tile_size);
			}  
		}
		
		// get the start data
		valT *data = new valT[ get_local_size_x(rhs.get_nb_tile_x()*tile_size)
		                     * get_local_size_y<tile_size>(rhs.get_nb_tile_y()*tile_size)
		                     * (all-1)
		                     ];
		for (int y=0; y<rhs.get_nb_tile_y(); ++y) {
			for (int x=0; x<rhs.get_nb_tile_x(); ++x) {
				if (is_local(x,y)) continue;
				
				int start = starts[get_node(x,y)];
				const valT * ptr = rhs.get_tile(x,y);
				for (int i=0; i<tile_size*tile_size; ++i) {
					data[start+i] = ptr[i];
				}
				starts[get_node(x,y)] += tile_size*tile_size;
			}
		}
		
		// copy local data into dst::matrix
		
		// start scatter on other nodes
		
		// free data
		delete []data;
		delete []starts;
		
		// TODO have the data freed locally
		for (int i=0; i<adabs::all; ++i) {
			if (i!=adabs::me) remote_free (remote_addrs[i]);
		}
			
		delete[] remote_addrs;


		// ok, this is realy ugly, but MPI_Scatter needs a non-const
		// send buffer pointer
		T& non_const_rhs = const_cast<T&>(rhs);
		
		// ok, now call MPI scatter
		// NOTE: We expect the data to be stored in one linear memory
		//       block so we do not have to copy it into a local mem
		//       block.
		// TODO: Comment this requirement somewhere or decide to remove
		//       it.
		/*MPI_Scatter (non_const_rhs.get_tile_unitialized(0,0),
		              get_local_size_x(lhs.get_size_x()),//*get_local_y(),
		              MPI_CHAR,
		              0,
		              get_local_size_x(lhs.get_size_x()),//*get_local_y(),
		              MPI_CHAR,
		              0,
		              MPI_COMM_WORLD
		            );*/
		
		std::cout << "distribution.scatter_root called" << std::endl;
	}
	
	template <typename T>
	static void scatter_receiver(T &rhs) {
		std::cout << "distribution.scatter_receiver called" << std::endl;
	}
};

#endif
