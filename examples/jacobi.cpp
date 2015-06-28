#include <omp.h>
#include <iostream>

#include "psam/matrix.hpp"

constexpr size_t TILE1 = 200;
constexpr size_t TILE2 = 50;

int main(int argc, char const *argv[]) {
	psam::memblock<size_t> mem(1000, {TILE1, TILE2});

#pragma omp parallel
	{
		const int id = omp_get_thread_num();
		if (id == 0) {
			size_t t = 0;
			while (t < mem.size) {
				{
					auto x = mem.start_write(t);
#pragma omp simd
					for (size_t i = 0; i < TILE1; ++i) {
						x.first[i] = i;
					}
					mem.end_write(x);
				}

				t += TILE1;
				{
					auto x = mem.start_write(t);
#pragma omp simd
					for (size_t i = 0; i < TILE2; ++i) {
						x.first[i] = i;
					}
					mem.end_write(x);
				}
				t += TILE2;
			}
		} else {
			int red = 0;
			for (size_t t = 0; t < mem.size; t += TILE1) {
				auto x = mem.read(t);
#pragma omp simd
				for (size_t i = 0; i < TILE1; ++i) {
					red += x[i];
				}
			}
			for (size_t t = 0; t < mem.size; t += TILE2) {
				auto x = mem.read(t);
#pragma omp simd
				for (size_t i = 0; i < TILE2; ++i) {
					red += x[i];
				}
			}
#pragma omp critical
			std::cout << "t: " << id << " red: " << red << std::endl;
		}
	}

	return 0;
}
