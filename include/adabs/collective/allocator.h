#pragma once

#include <queue>

#include "adabs/pgas_addr.h"
#include "adabs/tools/alignment.h"

namespace adabs {
namespace collective {

namespace pgas {
inline void remote_allocate_real(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                                gasnet_handlerarg_t arg1,
                                                gasnet_handlerarg_t arg2,
                                                gasnet_handlerarg_t arg3,
                                                gasnet_handlerarg_t arg4,
                                                gasnet_handlerarg_t arg5
                         );
inline void remote_malloc_real_reply(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                                    gasnet_handlerarg_t arg1,
                                                    gasnet_handlerarg_t arg2,
                                                    gasnet_handlerarg_t arg3
                             );
inline void remote_free_real(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                            gasnet_handlerarg_t arg1
                     );
inline void  add_to_stack(gasnet_token_t token, void *buf, size_t nbytes,
                                        gasnet_handlerarg_t arg0,
                                        gasnet_handlerarg_t arg1
                     );
}

namespace impl{
inline void* remote_allocate(const int node, const int num_objects, const int batch_size, const int sizeT, const int a);
inline void remote_free(const int node, void* ptr);
inline void* real_allocate(const int num_objects, const int batch_size, const int batch_mem_size, const int a);

extern std::queue<void*> memory_queue;
}


/**
 * This is our collective allocator, which will allocate memory on all
 * nodes. The memory will be connected with each other, so that it
 * looks like one uniform memory address. The memory allocation follows
 * this pattern:
 *    RRRRRFTTTTTTTTTTTTFTTTTTTTTTTTTFTTTTTTTTTTTT
 *    |-X-| |---num----| |---num----| |---num----|
 * whereas: - R are the addresses off this memory on the other node
 *          - X is the number of nodes this memory exists, which is
 *            currently identical to adabs::all
 *          - F is our flag indicating if data is available
 *          - T is the data
 *          - num indicates the batch size
 * NOTE: This allocator follows the STL concept, but is not expected
 *       to be STL conform. We may decide to bend (or even break)
 *       parts of what is expected from STL allocators. After all
 *       STL allocators are "weird" (see Effective STL).
 *       Currently missing:
 *       - non-const references
 *       - max_size()
 */

template <typename T>
class pgas_addr;

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
		allocate (num_objects, 1, localityHint);
	}
	static pointer allocate(size_type num_objects, size_type batch_size, const void* localityHint = 0);

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
allocator<T>::pointer allocator<T>::allocate(allocator<T>::size_type num_objects,
                                  allocator<T>::size_type batch_size,
                                  const void* localityHint) {
	using namespace adabs::tools;
	void* ptr;
	if (leader) {
		int a = tools::alignment<T>::val();
		if (a<sizeof(int)) a = sizeof(int);
		const size_t batch_mem_size  = sizeof(T) * batch_size + a;
		
		void **ptrs = new void*[all];
		
		for (int i=0; i<all; ++i) {
			if (i == me) {
				// local allocate
				ptrs[i] = impl::real_allocate (num_objects, batch_size, batch_mem_size, a);
			} else {
				// allocate on remote node
				ptrs[i] = adabs::collective::impl::remote_allocate(i, num_objects, batch_size, batch_mem_size, a);
			}
			
			//std::cout << "allocated on " << i << ": " << ptrs[i] << std::endl;
		}
		
		ptr = ptrs[me];
		
		// broadcast the addresses to all nodes and put the ptr on the stack
		for (int i=0; i<all; ++i) {
			if (i==me) {
				void** ptrptr = (void**)ptr;
				for (int i=0; i<all; ++i) {
					ptrptr[i] = ptrs[i];
				}
			} else {
				GASNET_CALL(gasnet_AMRequestLong2(i, adabs::impl::COLLECTIVE_ALLOC_BROADCAST,
				                                  ptrs, all*sizeof(T*), ptrs[i],
				                                  get_low(ptrs[i]),
				                                  get_high(ptrs[i])
								                 )
						   )
			}
		}
		
		for (int i=0; i<all; ++i) {
			//std::cout << "allocated on " << i << ": " << ptrs[i] << std::endl;
		}
		
		delete[] ptrs;
	} else {

		// get memory from the collective memory stack
		while (true) {
			bool end = false;
			// TODO OPTIMIZE ME!
			#pragma omp critical (queue)
			{
				if (!impl::memory_queue.empty()) {
					ptr = impl::memory_queue.front();
					impl::memory_queue.pop();
					end = true;
				}
			}
	
			if (end) break;
		}
		
		//std::cout << me << ": got " << ptr << std::endl;
	}
	
	return pointer(ptr, batch_size);
}

template <typename T>
void allocator<T>::deallocate(pointer &p) {
	// TODO: we could broadcast a 0 to all nodes so they will not sent
	//       us any information
	
	free(p._orig_ptr);
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
	// TODO
	throw "not implemented";
	
	//char* ptr = &x;
	//ptr -= sizeof(int);
	//return const_pointer((void*)ptr);
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

inline void* remote_allocate(const int node, const int num_objects, const int batch_size, const int sizeT, const int alignmentT) {
	using namespace adabs::tools;
	
	volatile long returnee;
	returnee = -1;
	
	 // start remote thread and allocate memory
	GASNET_CALL(gasnet_AMRequestShort6(node, adabs::impl::REMOTE_COLLECTIVE_ALLOC,
	                                   get_low(&returnee),
	                                   get_high(&returnee),
	                                   num_objects,
	                                   batch_size,
	                                   sizeT,
	                                   alignmentT
	                                  )
	           )
	 
	//wait until returnee != -1
	while (returnee == -1) {}

	return (void*)(returnee);
}


inline void remote_free(const int node, void* ptr) {
	using namespace adabs::tools;
	
	if (ptr == 0) return;
	
	GASNET_CALL(gasnet_AMRequestShort2(node, adabs::impl::REMOTE_COLLECTIVE_FREE,
	                                   get_low(ptr),
	                                   get_high(ptr)
	                                   )
	           )
}


inline void* real_allocate(const int num_objects, const int batch_size, const int batch_mem_size, const int alignmentT) {
	
	//std::cout << "real alloc parameter: " << num_objects << ", " << batch_size << ", " << batch_mem_size << ", " << alignmentT << std::endl;
	
	const size_t num_batch =   (num_objects % batch_size == 0)
	                         ? (num_objects / batch_size)
	                         : (num_objects / batch_size) + 1;
	
	const int pointer_alignment = alignmentT - ((adabs::all*sizeof(void*)) % alignmentT);
	const size_t mem_size = num_batch * batch_mem_size + adabs::all*sizeof(void*) + pointer_alignment;
	
	void *ptr = malloc (mem_size);
	
	void **init_ptr_1 = (void**)ptr;
	for (int i=0; i<adabs::all; ++i) {
		if (i!=me)
			*init_ptr_1 = 0;
		else
			*init_ptr_1 = ptr;
		
		++init_ptr_1;
	}
	
	char *init_ptr_2 = reinterpret_cast<char*>(init_ptr_1) + pointer_alignment;
	
	//std::cout << "malloc returned " << ptr << " to " << (void*)((char*)ptr + mem_size) << " - " << mem_size << std::endl;
	
	for (int i=0; i<num_batch; ++i) {
		init_ptr_2 += batch_mem_size - alignmentT;
		int *flag_ptr = reinterpret_cast<int*>(init_ptr_2);
		//std::cout << "write 0 to " << flag_ptr << std::endl;
		*flag_ptr = 0;
		init_ptr_2 += alignmentT;
	}
	
	return ptr;
}


}

namespace pgas {

inline void remote_allocate_real(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                                gasnet_handlerarg_t arg1,
                                                gasnet_handlerarg_t arg2,
                                                gasnet_handlerarg_t arg3,
                                                gasnet_handlerarg_t arg4,
                                                gasnet_handlerarg_t arg5
                       ) {
	using namespace adabs::tools;
	
	void* returnee = adabs::collective::impl::real_allocate(arg2, arg3, arg4, arg5);
	
	GASNET_CALL(gasnet_AMReplyShort4(token, adabs::impl::REMOTE_COLLECTIVE_ALLOC_REPLY,
	                                 arg0,
	                                 arg1,
	                                 get_low(returnee),
	                                 get_high(returnee)
	                                )
	           )
}

inline void remote_malloc_real_reply(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                                    gasnet_handlerarg_t arg1,
                                                    gasnet_handlerarg_t arg2,
                                                    gasnet_handlerarg_t arg3
                             ) {
	using namespace adabs::tools;
	long* local   = get_ptr<long> (arg0, arg1);
	void* remote  = get_ptr<void> (arg2, arg3);
	
	*local = reinterpret_cast<long>(remote);
}

inline void remote_free_real(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                                   gasnet_handlerarg_t arg1
                     ) {
	using namespace adabs::tools;
	void* ptr = get_ptr<void>(arg0, arg1);
	free (ptr);
}

inline void add_to_stack(gasnet_token_t token, void *buf, size_t nbytes,
                                        gasnet_handlerarg_t arg0,
                                        gasnet_handlerarg_t arg1
                     ) {
	using namespace adabs::tools;
	void* ptr = get_ptr<void>(arg0, arg1);
	
	#pragma omp critical (queue)
	adabs::collective::impl::memory_queue.push(ptr);
	
}

}

}
}
