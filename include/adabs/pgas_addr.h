#pragma once

#include <utility>

namespace adabs {

/**
 * A address in our pgas world
 */
template <typename T>
struct pgas_addr : public std::pair<int, T*> {
	pgas_addr() : std::pair<int, T*>() {}
	pgas_addr(const int i, T* t) : std::pair<int,  T*>(i, t) {}
	pgas_addr(const pgas_addr<T> &cpy) : std::pair<int,  T*>(cpy) {}
	
	int get_node() const {return this->first;}
	T* get_ptr() const {return this->second;} 
};

}
