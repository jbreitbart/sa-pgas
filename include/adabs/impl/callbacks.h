#pragma once

namespace adabs {

namespace impl {

extern gasnet_handlerentry_t *callbacks;

extern const int NUMBER_OF_CALLBACKS;

extern const int MEMCPY;

extern const int PGAS_ADDR_GET;
extern const int PGAS_ADDR_SET;
extern const int PGAS_ADDR_GET_UNINIT;
extern const int PGAS_ADDR_CHECK_GET_ALL;

extern const int COLLECTIVE_ALLOC_BROADCAST;
extern const int REMOTE_COLLECTIVE_ALLOC;
extern const int REMOTE_COLLECTIVE_FREE;
extern const int REMOTE_COLLECTIVE_ALLOC_REPLY;

extern const int COLLECTIVE_PGAS_ADDR_SET;

extern const int SET_RETURN_MARKER;
}

}
