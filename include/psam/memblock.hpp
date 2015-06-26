#pragma once

#include <cassert>
#include <cstdint>

namespace psam {

template <typename T, int X, int Y>
class memblock {
  private:
	enum class flag : std::int8_t { EMPTY, FULL };
	using T_arr = T[X * Y];

  private:
	T_arr _data;
	flag _flag = flag::EMPTY;

  public:
	static constexpr size_t size = X * Y;

  public:
	const T_arr &read() const {
		if (_flag == flag::EMPTY) {
			volatile flag const *const vol_flag = &_flag;
			while (*vol_flag == flag::EMPTY) {
			}
		}

		return _data;
	}

	T_arr &start_write() {
		assert(_flag == flag::EMPTY);
		return _data;
	}

	void end_write() {
		__sync_synchronize();
		_flag = flag::FULL;
	}
};
}
