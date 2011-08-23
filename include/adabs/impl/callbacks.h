#pragma once

namespace adabs {

namespace impl {

extern gasnet_handlerentry_t *callbacks;

extern const int NUMBER_OF_CALLBACKS;

/*extern const int MATRIX_BASE_INIT_GET;
extern const int MATRIX_BASE_INIT_SET;
extern const int MATRIX_BASE_SET;
extern const int MATRIX_BASE_GET;*/

//extern const int COLLECTIVE_VECTOR_GLOBAL_COM_SET;
//extern const int COLLECTIVE_VECTOR_SET;

/*extern const int DISTRIBUTED_MATRIX_BASE_DELETE;
extern const int DISTRIBUTED_MATRIX_BASE_REMOVE;
extern const int DISTRIBUTED_MATRIX_BASE_DELETE_ALL;
extern const int DISTRIBUTED_MATRIX_BASE_ENABLE_REUSE_ALL;
extern const int DISTRIBUTED_MATRIX_BASE_REUSE;
extern const int DISTRIBUTED_MATRIX_BASE_REUSE_REPLY;
extern const int DISTRIBUTED_MATRIX_BASE_ENABLE_REUSE_ALL_REPLY;
extern const int DISTRIBUTED_MATRIX_BASE_RESET_USE_FLAG;
extern const int DISTRIBUTED_MATRIX_BASE_SCATTER;*/

//extern const int MEMORY_MANAGEMENT_MALLOC;
//extern const int MEMORY_MANAGEMENT_MALLOC_REPLY;
//extern const int MEMORY_MANAGEMENT_FREE;

extern const int PGAS_ADDR_GET;
extern const int PGAS_ADDR_SET;

extern const int COLLECTIVE_ALLOC_BROADCAST;
extern const int REMOTE_COLLECTIVE_ALLOC;
extern const int REMOTE_COLLECTIVE_FREE;
extern const int REMOTE_COLLECTIVE_ALLOC_REPLY;

extern const int COLLECTIVE_PGAS_ADDR_SET;
}

}
