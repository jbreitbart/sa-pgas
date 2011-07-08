#pragma once

namespace adabs {

namespace tools {

/**
 * A simple tile struct, guaranteeing a alignment satisfying for
 * efficient SIMD load/store operations.
 */
template <typename T, int tile_size>
struct tile {
		// NOTE: keep flag after data, so we can copy the whole struct
		// by usding &data[0]
		T data[tile_size*tile_size] __attribute__((aligned (16)));
		volatile int flag;
	
		/*const T& operator[] (int i) const {return data[i];}
			  T& operator[] (int i)       {return data[i];}*/
	
		static int size() {return tile_size;}
	
		tile() {flag = 0;}
		~tile() {}
	
	private:
		// no copy, for now, I want to explicitly see the memory flow
		tile(const tile& x);
};

}

}
