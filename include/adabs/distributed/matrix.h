#pragma once

#include <vector>

#include "adabs/gasnet_config.h"
#include "adabs/pgas_addr.h"
#include "adabs/tools/tools.h"
#include "adabs/tools/tile.h"
#include "adabs/tools/ptr_divider.h"
#include "adabs/matrix.h"
#include "adabs/remote_matrix.h"
#include "adabs/collective/vector.h"

#include "adabs/distributed/matrix_base.h"

#include <mpi.h>

namespace adabs {

namespace distributed {

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
	
	template <typename T, typename T2>
	static void scatter_root(T const &rhs, const T2 &lhs) {
		//first check if all data is available
		for (int y=0; y<rhs.get_nb_tile_y(); ++y) {
			for (int x=0; x<rhs.get_nb_tile_x(); ++x) {
				rhs.get_tile(x,y);
			}
		}
		
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

/**
 * This is a tile based distributed matrix.
 *
 * Note: You should not inheriate from this matrix.
 * TODO comment me
 */
template <typename T, int tile_size, typename distribution>
class matrix : public matrix_base {
		/************************ TYPEDEFS ***************************/
	private:
		
	
		/************************ VARIABLES **************************/
	private:
		int delete_counter;
		int reuse_counter;
		const int _size_x;
		const int _size_y;
		const int _nb_tiles_x;
		const int _nb_tiles_y;
		
		adabs::collective::vector< pgas_addr< adabs::matrix<T, tile_size> > > _proxy_addrs;
		adabs::collective::vector< adabs::distributed::matrix_base* > _mes;
		
		adabs::matrix<T, tile_size> *_local_data;
		std::vector< adabs::remote_matrix<T, tile_size>* > _proxies;
				
		/********************* CON/DESTRUCTOS ************************/
	private:
		void create_proxies();
		
	public:
		/**
		 * Creates a matrix of size
		 * @param size_x * @param size_y
		 * . In case the size is not a multiply of the tile_size
		 * the size of the matrix is increased so it is a multiple of
		 * tile_size
		 */
		matrix (const int size_x, const int size_y)
		  : delete_counter(0), reuse_counter(0), _size_x(size_x), _size_y(size_y),
		    _nb_tiles_x((get_size_x()%tile_size == 0) ? (get_size_x()/tile_size) : (get_size_x()/tile_size+1)),
		    _nb_tiles_y((get_size_y()%tile_size == 0) ? (get_size_y()/tile_size) : (get_size_y()/tile_size+1)),
		    _proxy_addrs(adabs::all),
		    _mes(adabs::all),
		    _proxies(adabs::all)
		{
		    _local_data = new adabs::matrix<T, tile_size>(
		                         distribution::get_local_size_x(size_x), 
		                         distribution::template get_local_size_y<tile_size>(size_y) 
		                                                  );
			_proxy_addrs.set(adabs::me, _local_data->get_pgas_addr());
			_mes.set(adabs::me, this);
			create_proxies();
		}
		
		
		/**
		 * Desctructor, make sure to not delete the object before(!) all
		 * reads to that matrix are completed.
		 */
		virtual ~matrix() {
			using adabs::me;
			using adabs::all;
			for (int i=0; i<me; ++i) {
				delete _proxies[i];
			}
			for (int i=me+1; i<all; ++i) {
				delete _proxies[i];
			}
			
			delete _local_data;
		}
		
		/************************ FUNCTIONS **************************/
	private:

	public:
		/**
		 * Copies the values stored in
		 * @param ptr[0 ... tile_size*tile_size]
		 * to the tile with the coordinates
		 * @param x and @pscaaram y
		 * and marks the values as initialized. If @param *ptr is
		 * identical to a pointer returned by get_tile_unitialized
		 * no data will be copied.
		 */
		void set_tile(T const * restrict const ptr, const int x, const int y);
		
		/**
		 * Returns a pointer to the tile with the coordinated
		 * @param x and @param y
		 * . In case the values are not yet written to the matrix, the
		 * calling thread will sleep until the value is returned.
		 */
		T const* get_tile(const int x, const int y) const;
		
		/**
		 * Returns the pointer to the matrix internal tile with the
		 * coordinates
		 * @param x and  @param y
		 * so one can update the matrix in place. You must(!) still call
		 * set_tile() for this matrix tile!
		 */
		T* get_tile_unitialized(const int x, const int y);
		
		/**
		 * Returns the tile size
		 */
		static int get_tile_size() {return tile_size;}	

		/**
		 * Returns the number of tiles in x-dimension
		 */
		int get_nb_tile_x() const {
			return _nb_tiles_x;
		}
		
		/**
		 * Returns the number of tiles in y-dimension
		 */
		int get_nb_tile_y() const {
			return _nb_tiles_y;
		}

		/**
		 * Returns the size in x dimension
		 */
		int get_size_x() const {return _size_x;}
		
		/**
		 * Returns the size in x dimension
		 */
		int get_size_y() const {return _size_y;}
		
		void delete_all() {
			using namespace adabs::tools;
			using adabs::me;
			using adabs::all;
			
			// call real delete function on the remote matrix
			for (int i=0; i<all; ++i) {
				if (i!=me) {
					GASNET_CALL(gasnet_AMRequestShort2(i,
													   adabs::impl::DISTRIBUTED_MATRIX_BASE_DELETE,
													   get_low(_mes.get(i)),
													   get_high(_mes.get(i))
													  )
						       )
				}
			}
		
			delete this;
		}
		
		virtual bool remove(const bool local=true) {
			using namespace adabs::tools;
			using adabs::me;
			using adabs::all;
			
			
			if (me == 0) {
				bool returnee;
				#pragma omp critical
				{
					++delete_counter;
				
					returnee = delete_counter == all; 
				}
				
				// check if we were called on node 0 and everyone else has already called remove
				if (local && returnee) {
					delete_all();
				}
				
				return returnee;
			} else {
				// sent message @ node 0
				// node 0 increases counter
				GASNET_CALL(gasnet_AMRequestShort4(0,
										           adabs::impl::DISTRIBUTED_MATRIX_BASE_REMOVE,
										           get_low(_mes.get(0)),
										           get_high(_mes.get(0)),
										           get_low(_mes.get(me)),
										           get_high(_mes.get(me))
										          )
				           )
				while (get_delete_flag() == 0) {}
				
				// if the flag is 2, everyone has called remove and we must make
				// sure that the data gets deleted
				if (get_delete_flag() == 2) {
					delete_all();
				} 
			}
			
			return get_delete_flag() == 2;
		}
		
		void enable_reuse_all(const bool first_caller) {
			// 1. call enable_reuse_all on all nodes
			using namespace adabs::tools;
			using adabs::me;
			using adabs::all;
			
			// call real delete function on the remote matrix
			if (first_caller) {
				for (int i=0; i<all; ++i) {
					if (i!=me) {
						GASNET_CALL(gasnet_AMRequestShort4(i,
														   adabs::impl::DISTRIBUTED_MATRIX_BASE_ENABLE_REUSE_ALL,
														   get_low(_mes.get(i)),
														   get_high(_mes.get(i)),
														   get_low(this),
														   get_high(this)
														  )
								   )
					}
				}
			}
			
			// 2. mark caches as empty (do not delete data)
			for (int i=0; i<all; ++i) {
				if (i!=me) {
					_proxies[i] -> clear_cache();
				}
			}
			
			// 3. mark local data as unused
			_local_data -> reuse();
			
			// 4. wait until everyone is done (reply von DISTRIBUTED_MATRIX_BASE_ENABLE_REUSE_ALL)
			if (first_caller) {
				wait_until_remote_reuse_all();
			}
			
			// 5. set flag on all notes
			if (first_caller) {
				for (int i=0; i<all; ++i) {
					if (i!=me) {
						GASNET_CALL(gasnet_AMRequestShort2(i,
														   adabs::impl::DISTRIBUTED_MATRIX_BASE_RESET_USE_FLAG,
														   get_low(_mes.get(i)),
														   get_high(_mes.get(i))
														  )
								   )
					}
				}
				
				reset_reuse_all_remote_counter();
				
				set_use_flag_start();
			}
			
			reuse_counter = 0; // only relevant for me==0
		}
		
		bool enable_reuse(const bool local=true) {
			using namespace adabs::tools;
			using adabs::me;
			using adabs::all;
			//std::cout << me << ": renable reuse called" << std::endl;
			
			if (me == 0) {
				bool returnee;
				#pragma omp critical
				{ // TODO fix, use atomic op
					++reuse_counter;
				
					returnee = reuse_counter == all; 
				}
				
				// check if we were called on node 0 and everyone else has already called remove
				if (local && returnee) {
					enable_reuse_all(true);
				}
				
				return returnee;
			} else {
				// sent message @ node 0
				// node 0 increases counter
				GASNET_CALL(gasnet_AMRequestShort4(0,
										           adabs::impl::DISTRIBUTED_MATRIX_BASE_REUSE,
										           get_low(_mes.get(0)),
										           get_high(_mes.get(0)),
										           get_low(_mes.get(me)),
										           get_high(_mes.get(me))
										          )
				           )
				while (!resetted()) {}
				
				if (all_resetted()) {
					enable_reuse_all(true);
				} 
			}
			
			return all_resetted();
		}
		
		bool is_local(const int x, const int y) {
			return distribution::is_local(x,y);
		}
		
		/**
		 * This is far from perfect. Some things to consider:
		 * - I know this should not return void, but I am 
		 *   unsure how chaining of op= should look like, since this is a
		 *   collective operation.
		 * - We currently expect @param rhs to be a local
		 *   data structure. We should add compile time identifiers to
		 *   the classes so we know if they are local and may optimize
		 *   for this case. Most likely broken for non-local types.
		 * - We currently do not check if the data stored is compatible.
		 *   This may break really badly!!!
		 */
		template<typename T1>
		void operator=(const T1& rhs) {
			using namespace adabs::tools;
			
			using adabs::me;
			using adabs::all;
			
			std::cout << me << ": op= called" << std::endl;
			
			// start scatter operatiion on all nodes
			for (int i=0; i<me; ++i) {
				GASNET_CALL(gasnet_AMRequestShort2(i,
										           adabs::impl::DISTRIBUTED_MATRIX_BASE_SCATTER,
										           get_low(_mes.get(i)),
										           get_high(_mes.get(i))
										          )
						   )
			}
			for (int i=me+1; i<all; ++i) {
				GASNET_CALL(gasnet_AMRequestShort2(i,
										           adabs::impl::DISTRIBUTED_MATRIX_BASE_SCATTER,
										           get_low(_mes.get(i)),
										           get_high(_mes.get(i))
										          )
						   )
			}
			
			// ok, we are root and we start sending
			distribution::scatter_root(rhs, *_local_data);
			
		}
		
		void remote_scatter_caller() {
			using adabs::me;
			std::cout << me << ": remote scatter called" << std::endl;
			distribution::scatter_receiver(*_local_data);
		}
		
};

template <typename T, int tile_size, typename distribution>
void matrix<T, tile_size, distribution>::create_proxies() {
	using adabs::me;
	using adabs::all;
	for (int i=0; i<me; ++i) {
		_proxies[i] = new adabs::remote_matrix<T, tile_size>(_proxy_addrs.get(i));
	}
	for (int i=me+1; i<all; ++i) {
		_proxies[i] = new adabs::remote_matrix<T, tile_size>(_proxy_addrs.get(i));
	}
} 

template <typename T, int tile_size, typename distribution>
T* matrix<T, tile_size, distribution>::get_tile_unitialized(const int x, const int y) {
	const int local_x = distribution::get_local_x(x);
	const int local_y = distribution::get_local_y(y);
	
	if (distribution::is_local(x,y)) {
		//std::cout << me << ": local get tile uninit: " << x << ", " << y << ", " << local_x << ", " << local_y << std::endl;
		return _local_data->get_tile_unitialized(local_x,local_y);
	}
	
	//std::cout << me << ": remote get tile uninit: " << x << ", " << y << std::endl;
	return _proxies[distribution::get_node(x,y)]->get_tile_unitialized(local_x, local_y);
}

template <typename T, int tile_size, typename distribution>
void matrix<T, tile_size, distribution>::set_tile(T const * restrict const ptr, const int x, const int y) {
	const int local_x = distribution::get_local_x(x);
	const int local_y = distribution::get_local_y(y);
	
	if (distribution::is_local(x,y)) {
		//std::cout << me << ": local set tile: " << x << ", " << y << std::endl;
		_local_data->set_tile(ptr, local_x, local_y);
		return;
	}
	
	//#pragma omp critical
	//std::cout << me << ": remote set tile: " << x << ", " << y << ", " << ptr << std::endl;
	_proxies[distribution::get_node(x,y)]->set_tile(ptr, local_x, local_y);
}

template <typename T, int tile_size, typename distribution>
T const* matrix<T, tile_size, distribution>::get_tile(const int x, const int y) const {
	const int local_x = distribution::get_local_x(x);
	const int local_y = distribution::get_local_y(y);
	
	
	if (distribution::is_local(x,y)) return _local_data->get_tile(local_x, local_y);
	
	//std::cout << "remote" << std::endl;
	
	return _proxies[distribution::get_node(x,y)]->get_tile(local_x, local_y);
}

}

}
