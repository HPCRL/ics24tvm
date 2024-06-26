/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_scheduler/feature.h
 * \brief Feature extraction for the cost model.
 * We extract one feature vector per BufferStoreNode statement in a TIR Stmt,
 * so we call this feature as "per-store" feature.
 * The cost model also does prediction for each BufferStoreNode statement and aggregates
 * the predictions as the whole score for a TVM IR (Stmt).
 *
 * The feature specification is defined by `src/auto_scheduler/feature.cc:: FeatureSet`
 */

#ifndef TVM_AUTO_SCHEDULER_FEATURE_H_
#define TVM_AUTO_SCHEDULER_FEATURE_H_

#include <tvm/auto_scheduler/compute_dag.h>
#include <tvm/auto_scheduler/measure.h>
#include <tvm/tir/function.h>

#include <string>
#include <vector>

namespace tvm {
namespace auto_scheduler {


class splitMeta {
 public:
  int step_id;
  int adjust_stage_id;
  int problem_size;
  Iterator origin_itr;
  std::vector<int> tile_sizes;
  bool parallel;

  splitMeta(int step_id, int adjust_stage_id, int problem_size, int tile_len) {
    this->step_id = step_id;
    this->adjust_stage_id = adjust_stage_id;
    this->problem_size = problem_size;
  }
  ~splitMeta() {
    // std::cout << "delete class" << std::endl;
  }
  void add_tilesize(int i) { this->tile_sizes.push_back(i); }

  friend std::ostream& operator<<(std::ostream& os, const splitMeta& spm) {
    os << "stp : " << spm.step_id << " / " << spm.adjust_stage_id << "\n";
    os << "itr : " << spm.origin_itr->name << " / " << spm.problem_size << " / " << spm.parallel
       << "\n";
    os << "tile size len " << spm.tile_sizes.size() << "\n";
    os << "[ ";
    for (std::vector<int>::size_type i = 0; i < spm.tile_sizes.size(); ++i) {
      os << spm.tile_sizes[i] << ", ";
    }
    os << " ]";
    return os;
  }
};

struct TDx_access_info 
{
  std::string buffer_name;
  int buffer_touch_size;
  std::map<std::string, int> local_threadx_val_map;
};

std::tuple<int, int, float, float> extract_features(const SearchTask& task, State& state, std::vector<splitMeta*> v_splitMeta_info, std::vector<float> *features, std::vector<TDx_access_info> access_striding_info);


/*!
 * \brief Get per-store features from a TIR PrimFunc
 * \param func The input lowered TIR PrimFunc
 * \param cache_line_size The size of cache line in bytes
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param ret The returned feature vector
 * \param log_scale Should the outputs be scaled by log2(1+x).
 */
void GetPerStoreFeature(const PrimFunc& func, int cache_line_size, int max_n_bufs,
                        std::vector<float>* ret, bool log_scale = true);

void GetPerStoreOurFeature(const PrimFunc& func, int cache_line_size, int max_n_bufs,
                        std::vector<float>* ret, 
                        std::vector<size_t>* res,
                        tvm::Map<String, tvm::PrimExpr> gpu_params, std::vector<TDx_access_info>* access_striding_info, bool log_scale = true);
/*
 * \brief Get the names of elements in the feature vector. Use this for debug and inspection.
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param ret The returned names.
 */
void GetPerStoreFeatureName(int max_n_bufs, std::vector<std::string>* ret);

/*!
 * \brief Get per-store feature from states of the same task
 * \param states The input states
 * \param task The same search task for all states
 * \param skip_first_n_feature_extraction Skip feature extraction for the first n states
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param features The returned feature vector. The innermost vector contains the
 * feature vectors for all BufferStoreNode statements
 */
void GetPerStoreFeaturesFromStates(const Array<State>& states, const SearchTask& task,
                                   int skip_first_n_feature_extraction, int max_n_bufs,
                                   std::vector<std::vector<float> >* features);

void GetPerStoreFeaturesFromOneState(const State& states, const SearchTask& task, int max_n_bufs) ;

/*!
 * \brief Get per-store feature from states of different tasks
 * \param states The input states
 * \param tasks The search tasks corresponding to the input states
 * \param skip_first_n_feature_extraction Skip feature extraction for the first n states
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param features The returned feature vector. The innermost vector contains the
 * feature vectors for all BufferStoreNode statements
 */
void GetPerStoreFeaturesFromStates(const Array<State>& states, const std::vector<SearchTask>& tasks,
                                   int skip_first_n_feature_extraction, int max_n_bufs,
                                   std::vector<std::vector<float> >* features);

/*!
 * \brief Get per-store features from a log file
 * \param filename The name of log file
 * \param max_lines Only read the first n lines of the file
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param features The returned feature vector. The innermost vector contains the
 * feature vectors for all BufferStoreNode statements
 * \param normalized_throughputs The normalized throughputs for all states
 * \param task_ids The task ids for all states
 */
void GetPerStoreFeaturesFromFile(const std::string& filename, int max_lines, int max_n_bufs,
                                 std::vector<std::vector<float> >* features,
                                 std::vector<float>* normalized_throughputs,
                                 std::vector<int>* task_ids);

/*!
 * \brief Get per-store features from measurement input/result pairs
 * \param inputs The measurement inputs
 * \param results The measurement results
 * \param skip_first_n_feature_extraction Skip feature extraction for the first n measurement pairs
 * \param max_n_bufs The maximum number of extracted buffers for one statement
 * \param features The returned feature vector. The innermost vector contains the
 * feature vectors for all BufferStoreNode statements
 * \param normalized_throughputs The normalized throughputs for all states
 * \param task_ids The task ids for all states
 */
void GetPerStoreFeaturesFromMeasurePairs(const Array<MeasureInput>& inputs,
                                         const Array<MeasureResult>& results,
                                         int skip_first_n_feature_extraction, int max_n_bufs,
                                         std::vector<std::vector<float> >* features,
                                         std::vector<float>* normalized_throughputs,
                                         std::vector<int>* task_ids);

String GetScopeFromBufferName(String bfname);

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_FEATURE_H_
