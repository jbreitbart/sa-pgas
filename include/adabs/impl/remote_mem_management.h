#pragma once

#include <cstddef>
#include "adabs/adabs.h"

namespace adabs {

namespace impl {

void* remote_malloc(const int node, const std::size_t size);

void  remote_free(const int node, void* ptr);

namespace pgas {

void remote_malloc_real(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                              gasnet_handlerarg_t arg1,
                                              gasnet_handlerarg_t arg2
                       );

void remote_malloc_real_reply(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                                    gasnet_handlerarg_t arg1,
                                                    gasnet_handlerarg_t arg2,
                                                    gasnet_handlerarg_t arg3
                             );

void remote_free_real(gasnet_token_t token, gasnet_handlerarg_t arg0,
                                            gasnet_handlerarg_t arg1
                     );

}

}

}
