#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include <initializer_list>
#include <numeric>
#include <utility>
#include <iterator>

namespace psam {

template <typename T>
class memblock {
  private: /* CONSTANTS, TYPEDEFS */
	enum class flagT : std::int8_t { EMPTY, FULL };
	const size_t _offset_sum;

  public: /* CONSTANTS, TYPEDEFS */
	const size_t size;

  private: /* DATA */
	std::unique_ptr<T> _data;
	const std::vector<size_t> _offsets;
	std::vector<flagT> _flags;

  public: /* CONSTRUCTORS, DESTRUCTOR, ASSIGNMENTS */
	memblock() = delete;
	memblock(const memblock<T> &rhs) = delete;

	memblock(const size_t size_, std::initializer_list<size_t> offsets)
		: _offset_sum(std::accumulate(offsets.begin(), offsets.end(), size_t(0))), size(size_), _data(new T[size_]),
		  _offsets(offsets.begin(), offsets.end()) {
		// TODO optimize
		size_t temp = 0;
		size_t count = 0;
		while (temp < size) {
			++count;
			temp += _offsets.at(count % _offsets.size());
		}
		_flags.insert(_flags.end(), count, flagT::EMPTY);
	}

	memblock(memblock &&rhs) noexcept : _offset_sum(rhs._offset_sum),
										size(rhs.size),
										_data(std::move(rhs._data)),
										_offsets(std::move(rhs._offsets)),
										_flags(std::move(rhs._flags)) {
		assert(this != &rhs);
	}

	memblock<T> &operator=(const memblock<T> &rhs) = delete;

	memblock<T> &operator=(memblock<T> &&rhs) {
		assert(this != &rhs);

		_offset_sum = rhs._offset_sum;
		*const_cast<size_t *>(&size) = rhs.size;
		_data = std::move(rhs._data);
		_offsets = std::move(rhs._offsets);
		_flags = std::move(rhs._flags);
		return *this;
	}

  private: /* FUNCTIONS */
	flagT *get_flag(const size_t i) noexcept {
		return const_cast<flagT *>(static_cast<const memblock<T> *>(this)->get_flag(i));
	}

	const flagT *get_flag(const size_t i) const noexcept {
		size_t index = (i / _offset_sum);
		size_t temp = index * _offset_sum;
		for (auto o : _offsets) {
			if (temp + o < i) {
				temp += o;
				++index;
			} else {
				return &_flags[index];
			}
		}

		return &(_flags[index]) + index;
	}

  public: /* FUNCTIONS */
	const T *read(const size_t index) const noexcept {
		auto flag = get_flag(index);
		if (*flag == flagT::EMPTY) {
			volatile flagT const *const volptr_flag = flag;
			while (*volptr_flag == flagT::EMPTY) {
			}
		}

		assert(*flag == flagT::FULL);

		return _data.get() + index;
	}

	// TODO change the pointer somehow, should not be used outside this class
	std::pair<T *, flagT const *const> start_write(const size_t index) noexcept {
		auto flag = get_flag(index);
		assert(*flag == flagT::EMPTY);
		return std::make_pair(_data.get() + index, flag);
	}

	void end_write(std::pair<T *, flagT const *const> &d) noexcept {
		__sync_synchronize();
		*const_cast<flagT *>(d.second) = flagT::FULL;
	}

  private: /* FRIENDS, treat with car ;) */
#if 0
	friend std::ostream &operator<<(std::ostream &os, const memblock<T> &mem) {
		os << "size: " << mem.size << ", offsets: ";
		vector_to_stream(os, mem._offsets);
		os << ", flags: ";
		vector_to_stream(os, std::vector<int>(mem._flags));
		return os;
	}

	// TODO move somewhere elese
	template <class VT>
	static std::ostream &vector_to_stream(std::ostream &os, const std::vector<VT> &v) {
		os << "[";
		std::copy(v.begin(), v.end(), std::ostream_iterator<VT>(os, ", "));
		return os << "]";
	}
#endif
};
}
