#pragma once

#ifndef YIRAGE_BACKEND_MACA_ENABLED
#include "threadblock/blackwell_matmul.h"
#include "threadblock/blackwell_pipeline.h"
#endif
#include "threadblock/element_binary.h"
#include "threadblock/element_unary.h"
#include "threadblock/epilogues.h"
#include "threadblock/forloop_accum.h"
#ifndef YIRAGE_BACKEND_MACA_ENABLED
#include "threadblock/hopper_matmul.h"
#endif
#include "threadblock/input.h"
#include "threadblock/matmul.h"
#include "threadblock/output.h"
#include "threadblock/pipeline.h"
#include "threadblock/profiler.h"
#include "threadblock/reduction.h"
#include "threadblock/utils.h"