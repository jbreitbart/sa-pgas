#pragma once

#define CALLER 

#ifdef __CUDACC__
#define CALLER __device__ __host__
#endif


namespace adabs {

namespace tools {

template<typename T>
class alignment {
	private:
		struct align {
			char a;
			T t;
		};
	public:
		CALLER static int val() { 
			return sizeof(align) - sizeof(T);
		}
};

}

}
