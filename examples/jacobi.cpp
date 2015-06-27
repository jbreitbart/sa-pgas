#include <omp.h>
#include <iostream>

#include "psam/matrix.hpp"

int main(int argc, char const *argv[]) {
	psam::memblock<size_t> mem(1000, {250});

#pragma omp parallel
	{
		const int id = omp_get_thread_num();
		if (id == 0) {
			for (size_t t = 0; t < mem.size; t += 250) {
				auto x = mem.start_write(t);
#pragma omp simd
				for (size_t i = 0; i < 250; ++i) {
					x.first[i] = i;
				}
				mem.end_write(x);
			}
		} else {
			int red = 0;
			for (size_t t = 0; t < mem.size; t += 250) {
				auto x = mem.read(t);
#pragma omp simd
				for (size_t i = 0; i < 250; ++i) {
					red += x[i];
				}
			}
#pragma omp critical
			std::cout << "t: " << id << " red: " << red << std::endl;
		}
	}

	return 0;
}
