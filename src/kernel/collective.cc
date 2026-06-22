/* Copyright 2023-2024 CMU
 * Copyright 2025 Chen Xingqiang (YiRage Project)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernel/collective.h"
#include "kernel/device_memory_manager.h"
#include "kernel/graph.h"
#include "layout.h"
#include "utils/hash_utils.h"
#include <cassert>
#include <iostream>

namespace yirage {
namespace kernel {

// =============================================================================
// AllGather Implementation
// =============================================================================

DTensor Graph::all_gather(DTensor const &input, int num_participants, int gather_dim) {
  KNOperator *op = create_all_gather_op(input, num_participants, gather_dim);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

KNOperator *Graph::create_all_gather_op(DTensor const &input, 
                                        int num_participants, 
                                        int gather_dim) {
  KNAllGatherOp *op = new KNAllGatherOp(this, input, num_participants, gather_dim);
  return op;
}

KNAllGatherOp::KNAllGatherOp(Graph *_kgraph,
                             DTensor const &input,
                             int _num_participants,
                             int _gather_dim)
    : KNOperator(_kgraph, yirage::type::KN_ALLGATHER_OP, input),
      num_participants(_num_participants),
      gather_dim(_gather_dim) {
  
  // Output has expanded dimension at gather_dim
  DTensor output;
  output.num_dims = input.num_dims;
  for (int i = 0; i < input.num_dims; i++) {
    if (i == gather_dim) {
      output.dim[i] = input.dim[i] * num_participants;
    } else {
      output.dim[i] = input.dim[i];
    }
  }
  output.data_type = input.data_type;
  output.layout = input.layout;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  
  // Allocate memory for output
  kgraph->allocate(output);
  output_tensors.push_back(output);
}

KNAllGatherOp::~KNAllGatherOp() {
  kgraph->free(output_tensors[0]);
}

KNAllGatherOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"num_participants", num_participants},
              {"gather_dim", gather_dim}};
}

#if defined(YIRAGE_FINGERPRINT_USE_CPU) || defined(YIRAGE_FINGERPRINT_USE_ASCEND)
bool KNAllGatherOp::fingerprint(void) {
  return true;
}
#endif

// =============================================================================
// ReduceScatter Implementation
// =============================================================================

DTensor Graph::reduce_scatter(DTensor const &input, 
                              int num_participants, 
                              int scatter_dim,
                              yirage::type::CollectiveReduceOp reduce_op) {
  KNOperator *op = create_reduce_scatter_op(input, num_participants, scatter_dim, reduce_op);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

KNOperator *Graph::create_reduce_scatter_op(DTensor const &input,
                                            int num_participants,
                                            int scatter_dim,
                                            yirage::type::CollectiveReduceOp reduce_op) {
  KNReduceScatterOp *op = new KNReduceScatterOp(this, input, num_participants, 
                                                 scatter_dim, reduce_op);
  return op;
}

KNReduceScatterOp::KNReduceScatterOp(Graph *_kgraph,
                                     DTensor const &input,
                                     int _num_participants,
                                     int _scatter_dim,
                                     yirage::type::CollectiveReduceOp _reduce_op)
    : KNOperator(_kgraph, yirage::type::KN_REDUCE_SCATTER_OP, input),
      num_participants(_num_participants),
      scatter_dim(_scatter_dim),
      reduce_op(_reduce_op) {
  
  // Output has reduced dimension at scatter_dim
  DTensor output;
  output.num_dims = input.num_dims;
  for (int i = 0; i < input.num_dims; i++) {
    if (i == scatter_dim) {
      assert(input.dim[i] % num_participants == 0);
      output.dim[i] = input.dim[i] / num_participants;
    } else {
      output.dim[i] = input.dim[i];
    }
  }
  output.data_type = input.data_type;
  output.layout = input.layout;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  
  kgraph->allocate(output);
  output_tensors.push_back(output);
}

KNReduceScatterOp::~KNReduceScatterOp() {
  kgraph->free(output_tensors[0]);
}

KNReduceScatterOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"num_participants", num_participants},
              {"scatter_dim", scatter_dim},
              {"reduce_op", reduce_op}};
}

#if defined(YIRAGE_FINGERPRINT_USE_CPU) || defined(YIRAGE_FINGERPRINT_USE_ASCEND)
bool KNReduceScatterOp::fingerprint(void) {
  return true;
}
#endif

// =============================================================================
// Broadcast Implementation
// =============================================================================

DTensor Graph::broadcast(DTensor const &input, int num_participants, int source_rank) {
  KNOperator *op = create_broadcast_op(input, num_participants, source_rank);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

KNOperator *Graph::create_broadcast_op(DTensor const &input,
                                       int num_participants,
                                       int source_rank) {
  KNBroadcastOp *op = new KNBroadcastOp(this, input, num_participants, source_rank);
  return op;
}

KNBroadcastOp::KNBroadcastOp(Graph *_kgraph,
                             DTensor const &input,
                             int _num_participants,
                             int _source_rank)
    : KNOperator(_kgraph, yirage::type::KN_BROADCAST_OP, input),
      num_participants(_num_participants),
      source_rank(_source_rank) {
  
  // Output has same shape as input
  DTensor output;
  output = input;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  
  kgraph->allocate(output);
  output_tensors.push_back(output);
}

KNBroadcastOp::~KNBroadcastOp() {
  kgraph->free(output_tensors[0]);
}

KNBroadcastOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"num_participants", num_participants},
              {"source_rank", source_rank}};
}

#if defined(YIRAGE_FINGERPRINT_USE_CPU) || defined(YIRAGE_FINGERPRINT_USE_ASCEND)
bool KNBroadcastOp::fingerprint(void) {
  return true;
}
#endif

// =============================================================================
// P2P Send Implementation
// =============================================================================

void Graph::p2p_send(DTensor const &input, int dest_rank) {
  KNOperator *op = create_p2p_send_op(input, dest_rank);
  assert(op != nullptr);
  operators.push_back(op);
}

KNOperator *Graph::create_p2p_send_op(DTensor const &input, int dest_rank) {
  KNP2PSendOp *op = new KNP2PSendOp(this, input, dest_rank);
  return op;
}

KNP2PSendOp::KNP2PSendOp(Graph *_kgraph,
                         DTensor const &input,
                         int _dest_rank)
    : KNOperator(_kgraph, yirage::type::KN_P2P_SEND_OP, input),
      dest_rank(_dest_rank) {
  // Send has no output tensor
}

KNP2PSendOp::~KNP2PSendOp() {}

KNP2PSendOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"dest_rank", dest_rank}};
}

#if defined(YIRAGE_FINGERPRINT_USE_CPU) || defined(YIRAGE_FINGERPRINT_USE_ASCEND)
bool KNP2PSendOp::fingerprint(void) {
  return true;
}
#endif

// =============================================================================
// P2P Recv Implementation
// =============================================================================

DTensor Graph::p2p_recv(std::vector<int> const &dims, 
                        yirage::type::DataType dtype,
                        int source_rank) {
  KNOperator *op = create_p2p_recv_op(dims, dtype, source_rank);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

KNOperator *Graph::create_p2p_recv_op(std::vector<int> const &dims,
                                      yirage::type::DataType dtype,
                                      int source_rank) {
  KNP2PRecvOp *op = new KNP2PRecvOp(this, dims, dtype, source_rank);
  return op;
}

KNP2PRecvOp::KNP2PRecvOp(Graph *_kgraph,
                         std::vector<int> const &dims,
                         yirage::type::DataType dtype,
                         int _source_rank)
    : KNOperator(_kgraph, yirage::type::KN_P2P_RECV_OP),
      source_rank(_source_rank) {
  
  DTensor output;
  output.num_dims = dims.size();
  for (size_t i = 0; i < dims.size(); i++) {
    output.dim[i] = dims[i];
  }
  output.data_type = dtype;
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  
  kgraph->allocate(output);
  output_tensors.push_back(output);
}

KNP2PRecvOp::~KNP2PRecvOp() {
  kgraph->free(output_tensors[0]);
}

KNP2PRecvOp::operator json() const {
  return json{{"op_type", op_type},
              {"output_tensors", output_tensors},
              {"source_rank", source_rank}};
}

#if defined(YIRAGE_FINGERPRINT_USE_CPU) || defined(YIRAGE_FINGERPRINT_USE_ASCEND)
bool KNP2PRecvOp::fingerprint(void) {
  return true;
}
#endif

} // namespace kernel
} // namespace yirage
