#include <omp.h>
#include <iostream>

#include "psam/matrix.hpp"

int main(int argc, char const *argv[]) {
	psam::memblock<size_t, 100, 100> mem;

#pragma omp parallel
	{
		const int id = omp_get_thread_num();
		if (id == 0) {
			auto x = mem.start_write();
#pragma omp simd
			for (size_t i = 0; i < mem.size; ++i) {
				x[i] = i;
			}
			mem.end_write();
		} else {
			auto x = mem.read();
			int red = 0;
			for (size_t i = 0; i < mem.size; ++i) {
				red += x[i];
			}
#pragma omp critical
			std::cout << "t: " << id << " red: " << red << std::endl;
		}
	}

	return 0;
}
