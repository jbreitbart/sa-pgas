#include <omp.h>
#include <iostream>

#include "psam/matrix.hpp"

constexpr size_t TILE1 = 200;
constexpr size_t TILE2 = 50;
constexpr size_t SIZE = 1000;

int compute_reference() {
	size_t t = 0;
	int red = 0;
	while (t < SIZE) {
		for (size_t i = 0; i < TILE1; ++i) {
			red += i;
		}

		t += TILE1;
		for (size_t i = 0; i < TILE2; ++i) {
			red += i;
		}
		t += TILE2;
	}

	return red;
}

int main(int argc, char const *argv[]) {
	const int reference_output = compute_reference();

	while (true) {
		psam::memblock<size_t> mem(SIZE, {TILE1, TILE2});

#pragma omp parallel num_threads(8)
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
				size_t t = 0;
				while (t < mem.size) {
					{
						auto x = mem.read(t);

						for (size_t i = 0; i < TILE1; ++i) {
							red += x[i];
						}
						t += TILE1;
					}

					{
						auto x = mem.read(t);
						for (size_t i = 0; i < TILE2; ++i) {
							red += x[i];
						}
						t += TILE2;
					}
				}

				assert(red == reference_output);
			}
		}
	}
	return 0;
}
