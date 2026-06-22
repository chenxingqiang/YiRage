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

#pragma once

#include "kernel/operator.h"
#include "type.h"

namespace yirage {
namespace kernel {

// =============================================================================
// COMET Collective Operations
// =============================================================================
// These classes implement explicit collective operations following the COMET
// paper's representation for compound operation dataflows.
//
// Reference: "COMET: A Framework for Modeling Compound Operation Dataflows
// with Explicit Collectives" (Negi et al.)

/**
 * AllGather collective operation.
 * 
 * Gathers data from all participants and distributes the complete result
 * to all participants. Output size = input_size * num_participants.
 */
class KNAllGatherOp : public yirage::kernel::KNOperator {
public:
  KNAllGatherOp(Graph *_graph, 
                DTensor const &input, 
                int num_participants,
                int gather_dim);
  ~KNAllGatherOp();
  bool fingerprint(void) override;

  operator json() const override;

public:
  int num_participants;  // Number of devices participating
  int gather_dim;        // Dimension to gather along
};

void from_json(json const &j, KNAllGatherOp &op);

/**
 * ReduceScatter collective operation.
 * 
 * Reduces data across all participants and scatters the result.
 * Each participant gets a portion of the reduced result.
 * Output size = input_size / num_participants.
 */
class KNReduceScatterOp : public yirage::kernel::KNOperator {
public:
  KNReduceScatterOp(Graph *_graph,
                    DTensor const &input,
                    int num_participants,
                    int scatter_dim,
                    yirage::type::CollectiveReduceOp reduce_op);
  ~KNReduceScatterOp();
  bool fingerprint(void) override;

  operator json() const override;

public:
  int num_participants;
  int scatter_dim;
  yirage::type::CollectiveReduceOp reduce_op;
};

void from_json(json const &j, KNReduceScatterOp &op);

/**
 * Broadcast collective operation.
 * 
 * Broadcasts data from one source to all other participants.
 */
class KNBroadcastOp : public yirage::kernel::KNOperator {
public:
  KNBroadcastOp(Graph *_graph,
                DTensor const &input,
                int num_participants,
                int source_rank);
  ~KNBroadcastOp();
  bool fingerprint(void) override;

  operator json() const override;

public:
  int num_participants;
  int source_rank;  // Source device for broadcast
};

void from_json(json const &j, KNBroadcastOp &op);

/**
 * Point-to-point send operation.
 */
class KNP2PSendOp : public yirage::kernel::KNOperator {
public:
  KNP2PSendOp(Graph *_graph,
              DTensor const &input,
              int dest_rank);
  ~KNP2PSendOp();
  bool fingerprint(void) override;

  operator json() const override;

public:
  int dest_rank;
};

void from_json(json const &j, KNP2PSendOp &op);

/**
 * Point-to-point receive operation.
 */
class KNP2PRecvOp : public yirage::kernel::KNOperator {
public:
  KNP2PRecvOp(Graph *_graph,
              std::vector<int> const &dims,
              yirage::type::DataType dtype,
              int source_rank);
  ~KNP2PRecvOp();
  bool fingerprint(void) override;

  operator json() const override;

public:
  int source_rank;
};

void from_json(json const &j, KNP2PRecvOp &op);

} // namespace kernel
} // namespace yirage
