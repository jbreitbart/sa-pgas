#pragma once

#define GASNET_SEGMENT_EVERYTHING 1
#define GASNET_PAR
//#define GASNET_PARSYNC
//#define GASNET_CONDUIT_IBV 1

#define restrict __restrict__

#include <gasnet.h>

#include "adabs/impl/callbacks.h"
