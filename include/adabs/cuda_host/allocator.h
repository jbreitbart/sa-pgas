#pragma once

#include "adabs/cuda_host/pgas_addr.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace adabs {

namespace cuda_host {

template <typename T>
class pgas_addr;

namespace impl {
inline void* real_allocate(const int num_objects, const int batch_size, const int sizeT);
}

/**
 * This is our single assignment allocator, that is it creates all our
 * single assignment shared memory. It will allocate memory as follows
 *    FTTTTTTTTTTTFTTTTTTTTTTTFTTTTTTTTTTT
 *     |---num---| |---num---| |---num---|
 * whereas F is our flag indicating if data is available
 *         T is the data
 *         num indicates the batch size
 * NOTE: This allocator follows the STL concept, but is not expected
 *       to be STL conform. We may decide to bend (or even break)
 *       parts of what is expected from STL allocators. After all
 *       STL allocators are "weird" (see Effective STL)
 *       Currently missing:
 *       - non-const references
 *       - max_size()
 */


template <typename T>
struct allocator;

// specialize for void, since we can't have void&
// not sure if we really need this...
template <>
struct allocator<void> {
	typedef pgas_addr<void>       pointer;
	typedef const pgas_addr<void> const_pointer;
	
	typedef void value_type;
	template <class U> struct rebind { typedef allocator<U> other; };
};

    
template <typename T>
struct allocator {
	/***************** TYPEDEFS ***********************/
	typedef size_t             size_type;
	//typedef ptrdiff_t          difference_type;
	typedef pgas_addr<T>       pointer;
	typedef const pgas_addr<T> const_pointer;
	typedef T                  value_type;
	//typedef T&        reference;
	typedef const T&  const_reference;
	
	// allows to rebind this allocator to a different type
	template <typename U> struct rebind { typedef allocator<U> other; };

	/***************** CONSTRUCTORS ********************/
	allocator() throw() {}
	allocator(const allocator&) throw() {}
	template <class U> allocator(const allocator<U>&) throw() {}
	~allocator() throw() {}
	
	/****************** ALLOCATE **********************/
	static pointer allocate(size_type num_objects, const void* localityHint = 0) {
		void* ptr = allocate (num_objects, 1, localityHint);
		return pointer(ptr, 1);
	}
	static pointer allocate(size_type num_objects, size_type batch_size, const void* localityHint = 0) {
		void* ptr = impl::real_allocate(num_objects, batch_size, sizeof(T));
		return pointer(ptr, batch_size);
	}

	/****************** DEALLOCATE **********************/
	static void deallocate(pointer &p, size_type n) {
		deallocate(p);
	}
	static void deallocate(pointer &p);

	/****************** CONSTRUCT **********************/
	static void construct(pointer p, const T& val);
	static void destroy(pointer p);

	/****************** ADDRESS **********************/
	static const_pointer address(const_reference x);
	
};

template <typename T>
void allocator<T>::deallocate(pointer &p) {
	assert(p.is_local());
	
	cudaFreeHost(p._orig_ptr);
}

template <typename T>
void allocator<T>::construct(pointer p, const T& val) {
	// TODO
	throw "not implemented";
	
	p->T(val);
}

template <typename T>
void allocator<T>::destroy(pointer p) {
	// TODO
	throw "not implemented";
	
	p->~T();
}

template <typename T>
allocator<T>::const_pointer allocator<T>::address(const_reference x) {
	char* ptr = &x;
	ptr -= sizeof(int);
	return const_pointer((void*)ptr);
}

template <class T1, class T2>
bool operator==(const allocator<T1>&, const allocator<T2>&) throw() {
	return true;
}

template <class T1, class T2>
bool operator!=(const allocator<T1>&, const allocator<T2>&) throw() {
	return false;
}

namespace impl {

inline void* real_allocate(const int num_objects, const int batch_size, const int sizeT) {
	const size_t batch_mem_size = sizeT * num_objects + sizeof(int);
	
	const size_t num_batch =   (num_objects % batch_size == 0)
	                         ? (num_objects / batch_size)
	                         : (num_objects / batch_size) + 1;
	
	const size_t mem_size = num_batch * batch_mem_size;
	
	void *ptr;
	cudaError_t error = cudaHostAlloc(&ptr, mem_size, cudaHostAllocDefault);//cudaMallocHost (&ptr, mem_size);
	
	if(error != cudaSuccess) {
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	//void *ptr = malloc (mem_size);
	
	char *init_ptr = reinterpret_cast<char*>(ptr);
	
	for (int i=0; i<num_batch; ++i) {
		int *flag_ptr = reinterpret_cast<int*>(init_ptr);
		*flag_ptr = 0;
		//std::cout << "write 0 to " << flag_ptr << std::endl;
		init_ptr += batch_mem_size;
	}
	
	return ptr;
}

}

}

}

