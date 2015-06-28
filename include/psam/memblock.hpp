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
	/**** CONSTANTS, TYPEDEFS ****/
  private:
	// currently one flag per cache line. Maybe overkill.
	// TODO think about it
	enum class flagT : std::int8_t { EMPTY, FULL } __attribute__((aligned(64)));
	const size_t _offset_sum;

  public:
	const size_t size;

	/**** DATA ****/
  private:
	std::unique_ptr<T[]> _data;
	const std::vector<size_t> _offsets;
	std::vector<flagT> _flags;

	/**** CONSTRUCTORS, DESTRUCTOR, ASSIGNMENTS ****/
  public:
	memblock() = delete;
	memblock(const memblock<T> &rhs) = delete;
	memblock<T> &operator=(const memblock<T> &rhs) = delete;

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

	memblock<T> &operator=(memblock<T> &&rhs) noexcept {
		assert(this != &rhs);

		_offset_sum = rhs._offset_sum;
		*const_cast<size_t *>(&size) = rhs.size;
		_data = std::move(rhs._data);
		_offsets = std::move(rhs._offsets);
		_flags = std::move(rhs._flags);
		return *this;
	}

	/**** FUNCTIONS ****/
  private:
	flagT *get_flag(const size_t i) noexcept {
		return const_cast<flagT *>(static_cast<const memblock<T> *>(this)->get_flag(i));
	}

	const flagT *get_flag(const size_t i) const noexcept {
		size_t index = (i / _offset_sum) * _offsets.size();
		size_t temp = (i / _offset_sum) * _offset_sum;

		for (auto o : _offsets) {
			if (temp + o <= i) {
				temp += o;
				++index;
			} else {
				return &_flags[index];
			}
		}

		return &(_flags[index]) + index;
	}

	/**** FUNCTIONS ****/
  public:
	T const *const read(const size_t index) const noexcept {
		assert(index < size);
		volatile flagT const *const volptr_flag = get_flag(index);

		while (*volptr_flag != flagT::FULL) {
		}

		return _data.get() + index;
	}

	// TODO change the pointer somehow, should not be used outside this class
	std::pair<T *, void const *const> start_write(const size_t index) noexcept {
		assert(index < size);

		auto flag = get_flag(index);
		assert(*flag == flagT::EMPTY);

		return std::make_pair(_data.get() + index, reinterpret_cast<void const *const>(flag));
	}

	// TODO change the pointer somehow, should not be used outside this class
	void end_write(std::pair<T *, void const *const> &d) noexcept {
		__sync_synchronize();
		*reinterpret_cast<volatile flagT *>(const_cast<void *>(d.second)) = flagT::FULL;
	}

	/**** FRIENDS, treat with care ;) ****/
  private:
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
