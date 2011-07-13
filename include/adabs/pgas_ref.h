#pragma once

#include "adabs/adabs.h"
#include "adabs/pgas_addr.h"

namespace adabs {

namespace pgas {

// read at node @param node on pointer @param ptr @param size bytes
void remote_read(const int node, const void *ptr, void* returnee, const int size);

// writes data to node @param node on address @param ptr
void remote_write(const int node, void *ptr, const void* src, const int size);

}


/**
 * A global reference
 */
template <typename T>
struct pgas_ref {
		// TODO cache value?
		
		pgas_addr<T> addr;
		
		pgas_ref (const pgas_addr<T>& _addr) : addr(_addr) { }
		pgas_ref (const int node, T* t) : addr(node, t) { }
		pgas_ref (const pgas_ref<T>& _ref) : addr(_ref.addr) {}
		
		// this is called when data is just read
		operator T() const {
			if (addr.get_node() == adabs::me) return *addr.get_ptr();
					
			T returnee;
			
			pgas::remote_read(addr.get_node(), (const void*)addr.get_ptr(), (void*)&returnee, sizeof(T));
			
			return returnee;
		}
		
		// this is called when data is written to
		// Note that this will do a byte-wise copy
		// this is a local to remote assignment
		pgas_ref<T>& operator=(const          T  &rhs) {
			if (rhs == *this) return *this;
			
			if (addr.get_node() == adabs::me) {
				*addr.get_ptr() = rhs;
			} else {
				pgas::remote_write(addr.get_node(), (void*)addr.get_ptr(), (const void*)&rhs, sizeof(T));
			}
			
			return *this;
		}
		
		bool operator==(const pgas_ref<T> &rhs) const {
			return (addr == rhs.addr);
		}
		
	private:
		pgas_ref() {}
};


}
#if 0
		// the first one is a (possible) remote to remote assignment
		
		/*pgas_ref<T>& operator=(const pgas_ref<T> &rhs) {
			if (rhs == *this) return *this;
			T val;
			
			if (rhs.addr.get_node() == adabs::me) {
				val = *rhs.addr.get_ptr();
			} else {
				pgas::remote_read(rhs.addr.get_node(), rhs.addr.get_ptr(), (void*)&val, sizeof(T));
			}
			
			if (addr.get_node() == adabs::me) {
				*addr.get_ptr() = val;
			} else {
				pgas::remote_write(addr.get_node(), addr.get_ptr(), (void*)&val, sizeof(T));
			}
			
			return *this;
		}*/
#endif

