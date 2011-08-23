#pragma once

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
		static int val() { 
			return sizeof(align) - sizeof(T);
		}
};

}

}
