#pragma once

#include <stdint.h>

namespace adabs {

namespace tools {

namespace ptr_divider_impl {

struct lowhighT {
	int32_t low;
	int32_t high;
};

template<typename T>
union dataT {
	T* ptr;
	lowhighT lh;
	
	dataT(T* p) {
		ptr = p;
	}
	
	dataT(int32_t low, int32_t high) {
		lh.low = low;
		lh.high = high;
	}
};

}

template<typename T>
int32_t get_low(T* ptr) {
	ptr_divider_impl::dataT<T> a(ptr);
	return a.lh.low;
}

template<typename T>
int32_t get_high(T* ptr) {
	ptr_divider_impl::dataT<T> a(ptr);
	return a.lh.high;
}

template<typename T>
T* get_ptr(int32_t low, int32_t high) {
	ptr_divider_impl::dataT<T> a(low,high);
	return a.ptr;
}

}

}

