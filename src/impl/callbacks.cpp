#pragma once

#include "adabs/gasnet_config.h"

namespace adabs {

namespace impl {

gasnet_handlerentry_t *callbacks;

const int MATRIX_BASE_INIT_GET = 128;
const int MATRIX_BASE_INIT_SET = 129;
const int COLLECTIVE_VECTOR_GLOBAL_COM_SET = 130;
const int COLLECTIVE_VECTOR_SET = 131;
const int MATRIX_BASE_SET = 132;
const int MATRIX_BASE_GET = 133;
const int DISTRIBUTED_MATRIX_BASE_DELETE = 134;
const int DISTRIBUTED_MATRIX_BASE_REMOVE = 135;
const int DISTRIBUTED_MATRIX_BASE_DELETE_ALL = 136;
const int DISTRIBUTED_MATRIX_BASE_ENABLE_REUSE_ALL = 137;
const int DISTRIBUTED_MATRIX_BASE_REUSE = 138;
const int DISTRIBUTED_MATRIX_BASE_REUSE_REPLY = 139;
const int DISTRIBUTED_MATRIX_BASE_ENABLE_REUSE_ALL_REPLY = 140;
const int DISTRIBUTED_MATRIX_BASE_RESET_USE_FLAG = 141;

const int NUMBER_OF_CALLBACKS = 14;

}

}