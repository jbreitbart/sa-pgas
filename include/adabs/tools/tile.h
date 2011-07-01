#pragma once

namespace adabs {

namespace tools {

/**
 * A simple tile struct, guaranteeing a alignment satisfying for
 * efficient SIMD load/store operations.
 */
template <typename T, int tile_size>
struct tile {
		T data[tile_size*tile_size] __attribute__((aligned (16)));
	
		/*const T& operator[] (int i) const {return data[i];}
			  T& operator[] (int i)       {return data[i];}*/
	
		static int size() {return tile_size;}
	
		tile() {}
		~tile() {}
	
	private:
		// no copy, for now, I want to explicitly see the memory flow
		tile(const tile& x);
};

}

}
