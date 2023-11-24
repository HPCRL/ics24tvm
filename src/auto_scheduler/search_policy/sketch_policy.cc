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
 * \file auto_scheduler/search_policy/sketch_search_policy.h
 * \brief The search policy that searches in a hierarchical search space defined by sketches.
 * The policy randomly samples programs from the space defined by sketches
 * and use evolutionary search to fine-tune them.
 */

#include "sketch_policy.h"
#include <tvm/auto_scheduler/feature.h>

#include <tvm/runtime/registry.h>
#include <tvm/support/parallel_for.h>

#include <algorithm>
#include <iomanip>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>

#include "sketch_policy_rules.h"
#include "sketch_analysis.h"
#include <assert.h>
#include <climits>

namespace tvm {
namespace auto_scheduler {

/********** Sketch generation rules **********/
static RuleSkipStage rule_skip_stage;
static RuleAlwaysInline rule_always_inline;
static RuleMultiLevelTiling rule_multi_level_tiling;
static RuleMultiLevelTilingWithFusion rule_multi_level_tiling_with_fusion;
static RuleAddCacheRead rule_add_cache_read_stage;
static RuleAddCacheWrite rule_add_cache_write_stage;
static RuleAddRfactor rule_add_rfactor;
static RuleCrossThreadReduction rule_cross_thread_reduction;
static RuleSimplifyComputeWithConstTensor rule_simplify_compute_with_const_tensor;
static RuleSpecialComputeLocationGPU rule_special_compute_location_gpu;

/********** Init population rules **********/
static InitFillTileSize init_fill_tile_size;
static InitFillTileSizeUnique init_fill_tile_size_unique;
static InitChangeComputeLocation init_change_compute_location;
static InitParallel init_parallel;
static InitUnroll init_unroll;
static InitVectorization init_vectorization;
static InitThreadBind init_thread_bind;

/********** Sketch policy **********/
TVM_REGISTER_NODE_TYPE(SketchPolicyNode);

SketchPolicy::SketchPolicy(SearchTask task, CostModel program_cost_model,
                           Map<String, ObjectRef> params, int seed, int verbose,
                           Optional<Array<SearchCallback>> init_search_callbacks) {
  auto node = make_object<SketchPolicyNode>();
  node->search_task = std::move(task);
  node->program_cost_model = std::move(program_cost_model);
  node->rand_gen = std::mt19937(seed);
  node->params = std::move(params);
  node->verbose = verbose;
  node->sample_init_min_pop_ =
      GetIntParam(node->params, SketchParamKey::SampleInitPopulation::min_population);

  if (init_search_callbacks) {
    PrintTitle("Call init-search callbacks", verbose);
    // Candidates:
    // - auto_scheduler.PreloadMeasuredStates: Load already measured states to
    //   `measured_states_set_`, `measured_states_vector_` and `measured_states_throughputs_`.
    // - auto_scheduler.PreloadCustomSketchRule: Add user custom sketch rules to `sketch_rules`,
    //   these rules will be processed prior to the default rules.
    node->RunCallbacks(init_search_callbacks.value());
  }

  // NOTE: There are strong dependency among the rules below,
  // so the order to push them into the vector should be considered carefully.
  if (IsCPUTask(node->search_task)) {
    // Sketch Generation Rules
    node->sketch_rules.push_back(&rule_always_inline);
    node->sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
    node->sketch_rules.push_back(&rule_add_rfactor);
    node->sketch_rules.push_back(&rule_add_cache_write_stage);
    node->sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
    node->sketch_rules.push_back(&rule_multi_level_tiling);
    node->sketch_rules.push_back(&rule_skip_stage);

    // Initial Population Generation Rules
    node->init_rules.push_back(&init_fill_tile_size);
    node->init_rules.push_back(&init_change_compute_location);
    node->init_rules.push_back(&init_parallel);
    node->init_rules.push_back(&init_unroll);
    node->init_rules.push_back(&init_vectorization);

    // Mutation Rules for Evolutionary Search
    node->mutation_rules.push_back(std::make_shared<MutateTileSize>(0.90));
    node->mutation_rules.push_back(std::make_shared<MutateAutoUnroll>(0.04));
    node->mutation_rules.push_back(std::make_shared<MutateComputeLocation>(0.05));
    node->mutation_rules.push_back(std::make_shared<MutateParallel>(0.01));
  } else if (IsGPUTask(node->search_task)) {
    // Sketch Generation Rules
    if (node->search_task->target->GetAttr<String>("device", "") == "mali") {
      node->sketch_rules.push_back(&rule_always_inline);
      node->sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
      node->sketch_rules.push_back(&rule_add_rfactor);
      node->sketch_rules.push_back(&rule_add_cache_write_stage);
      node->sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
      node->sketch_rules.push_back(&rule_multi_level_tiling);
      node->sketch_rules.push_back(&rule_skip_stage);
    } else {
      node->sketch_rules.push_back(&rule_add_cache_read_stage);
      node->sketch_rules.push_back(&rule_special_compute_location_gpu);
      node->sketch_rules.push_back(&rule_always_inline);
      node->sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
      node->sketch_rules.push_back(&rule_cross_thread_reduction);
      node->sketch_rules.push_back(&rule_add_cache_write_stage);
      node->sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
      node->sketch_rules.push_back(&rule_multi_level_tiling);
      node->sketch_rules.push_back(&rule_skip_stage);
    }

    // Initial Population Generation Rules
    node->init_rules.push_back(&init_fill_tile_size);
    node->init_rules.push_back(&init_thread_bind);
    node->init_rules.push_back(&init_unroll);

    if (node->search_task->target->GetAttr<String>("device", "") == "mali") {
      node->init_rules.push_back(&init_vectorization);
    }

    // Mutation Rules for Evolutionary Search
    node->mutation_rules.push_back(std::make_shared<MutateTileSize>(0.90));
    node->mutation_rules.push_back(std::make_shared<MutateAutoUnroll>(0.10));
  } else {
    LOG(FATAL) << "No default sketch rules for target: " << task->target;
  }

  data_ = std::move(node);
}

std::vector<splitMeta*> SketchPolicyNode::GenerateSplitMeta(SketchPolicyNode* policy, State state){
  // const State& init_state = task->compute_dag->init_state;
 
  // Extract all SplitStep
  // We need to adjust stage id of split_step since new added stages (cache read, write) changes the
  // id.
  std::map<int, int> split_step_ids_stage_id_shift_map;
  // reverse traverse
  int shift = 0;
  for (size_t i = state->transform_steps.size() - 1; i > 0; i--) {
    if (state->transform_steps[i].as<CacheWriteStepNode>() != nullptr ||
        state->transform_steps[i].as<CacheReadStepNode>() != nullptr) {
      shift += 1;
    }
    if (auto ps = state->transform_steps[i].as<SplitStepNode>()) {
      if (!ps->extent.defined() || !ps->extent.value()->IsInstance<IntImmNode>()) {
        continue;
      }
      split_step_ids_stage_id_shift_map[i] = shift;
    }
  }
  std::vector<splitMeta*> v_splitMeta_info;

  // No split node; No tile size could be changes.
  if (split_step_ids_stage_id_shift_map.empty()) {
    return v_splitMeta_info;
  }

  
  for (auto itr = split_step_ids_stage_id_shift_map.cbegin();
       itr != split_step_ids_stage_id_shift_map.cend();) {
    int step_id = itr->first;
    auto ps = state->transform_steps[step_id].as<SplitStepNode>();
    int orgin_stage_id = ps->stage_id;
    int adjust_stage_id = orgin_stage_id + itr->second;
    auto stg = state->stages[adjust_stage_id];

    if (stg->op->name.find("shared") != std::string::npos) {
      // save remove CHR stage splitnode
      split_step_ids_stage_id_shift_map.erase(itr++);
    } else {
      int extent = GetIntImm(ps->extent.value());
      splitMeta* spm = new splitMeta(step_id, adjust_stage_id, extent, ps->lengths.size() + 1);
      v_splitMeta_info.push_back(spm);
      ++itr;
    }
  }
  return v_splitMeta_info;
}

State SketchPolicyNode::Search(int n_trials, int early_stopping, int num_measure_per_iter,
                               ProgramMeasurer measurer) {
  num_measure_per_iter_ = num_measure_per_iter;

  if (n_trials <= 1) {
    // No measurement is allowed
    const Array<State>& best_states = SearchOneRound(0);
    ICHECK_GT(best_states.size(), 0);
    return best_states[0];
  } else {
    int num_random =
        static_cast<int>(GetDoubleParam(params, SketchParamKey::eps_greedy) * num_measure_per_iter);
    early_stopping = early_stopping < 0 ? std::numeric_limits<int>::max() >> 1 : early_stopping;
    measurer->Reset();

    int ct = 0;
    int empty_retry_count = GetIntParam(params, SketchParamKey::empty_retry_count);
    Array<State> best_states, next_states;
    Array<MeasureInput> inputs;
    Array<MeasureResult> results;
    Array<State> local_min_best_states, track_path;
    // record all local min states
    Array<State> local_min_set, initStatesForModel;

    bool firsttime_random = true;
    int max_num_for_measure = 16;
    num_failed_local_search_ = 0;
    int init_num = 16;

    // generate a model based on random sampling and measure them
    if (ct == 0){ // just run it at the first time
      if (sketch_cache_.empty()) {
        sketch_cache_ = GenerateSketches();
      }
      initStatesForModel = SampleInitPopulation(sketch_cache_);
      initStatesForModel = search_task->compute_dag.InferBound(initStatesForModel);
      // sample 64 to update the model
      inputs = PackStateForModel(initStatesForModel, initStatesForModel.size());

      int model_age = 1;
      std::vector<float> r_scores;
      if (!inputs.empty()) {
        r_scores.reserve(inputs.size());
        Array<State> measure_states;
        measure_states.reserve(inputs.size());
        for (size_t j = 0; j < inputs.size(); ++j) {
            State tmp = inputs[j].get()->state;
            measure_states.push_back(tmp);
        }
        program_cost_model->Predict(search_task, measure_states, &r_scores);

        // use xMeasure to avoid write into the json log
        results = measurer->xMeasure(search_task, GetRef<SearchPolicy>(this), inputs, r_scores, model_age);

        auto t_begin = std::chrono::high_resolution_clock::now();

        // Retrain the cost model before the next search round
        PrintTitle("Model trained for the neighbor search", verbose);
        program_cost_model->Update(inputs, results);

        PrintTimeElapsed(t_begin, "training", verbose);
      }
    }
    
    while (ct < n_trials) {
      // create new predict based search
      local_min_best_states = SearchOneRoundPruePredict(init_num, &next_states, firsttime_random);
      // std::cout << "Num of local min got: #" << local_min_set.size() << std::endl;
      // std::cout << "Num of next_states: #" << next_states.size() << std::endl;
      if (next_states.empty()){
        if (local_min_set.empty()){// No base nodes, no local min in stashed local_min_set, consider as fail
          num_failed_local_search_ += 1;
        }
        // NO more neighbour to explore, resample.
        firsttime_random = true;
        firstround = true;
      }
      else{
        firsttime_random = false;
      }

      int count_duplicate = 0;
      if (!local_min_best_states.empty()){ // if we get some local min states, add them to local_min_set
        for (auto localmin : local_min_best_states){
          local_min_set.push_back(localmin);
        }
      }

      if (local_min_set.size() + ct >= n_trials || local_min_set.size() >= max_num_for_measure){
        // once local_min_set is large enough, measure them
        local_min_set = search_task->compute_dag.InferBound(local_min_set);
        // TODO: (Chendi)here local_min_set has no duplicate, should be safe to directly measure, remove measured_states_set_ check
        inputs = PackState(local_min_set, n_trials - ct);
        local_min_set.clear();
        if (!inputs.empty()) {
          // Measure candidate states
          PrintTitle("Measure Local MIN", verbose);
          results = measurer->Measure(search_task, GetRef<SearchPolicy>(this), inputs);
          // reset empty_retry_count
          empty_retry_count = GetIntParam(params, SketchParamKey::empty_retry_count);

          auto t_begin = std::chrono::high_resolution_clock::now();

          // Retrain the cost model before the next search round
          PrintTitle("****** Train cost model", verbose);
          program_cost_model->Update(inputs, results);

          PrintTimeElapsed(t_begin, "training", verbose);
        }
        // std::cout << "****** Number of measurement: " << inputs.size() << std::endl;
        ct += inputs.size();
      }

      // Stop if hit too many same local min states
      // std::cout << "Rest sequential failed count before early stop : " << empty_retry_count << std::endl;
      if (num_failed_local_search_ != 0){
        // check if too many sequantial failed or local_min_set has no local min states to measure, early stop
        if (empty_retry_count - num_failed_local_search_ > 0 || local_min_set.size() > 0){
          // in case we can't get enough states in local_min_set, decrease the measure threshold to at lease measure some states
          empty_retry_count -= num_failed_local_search_;
          num_failed_local_search_ = 0;
          continue;
        } else {
          StdCout(verbose) << "Hitting too many same local min states, stop" << std::endl;
          break;
        }
      }
    } //End of while loop
    

    PrintTitle("Done", verbose);

    //think return state;
    return measurer->best_state[search_task->workload_key];
  }
}

std::pair<Array<MeasureInput>, Array<MeasureResult>> SketchPolicyNode::ContinueSearchOneRound(
    int num_measure, ProgramMeasurer measurer) {
  num_measure_per_iter_ = num_measure;

  Array<State> best_states, random_states;
  Array<MeasureInput> inputs;
  Array<MeasureResult> results;
  int num_random = static_cast<int>(GetDoubleParam(params, "eps_greedy") * num_measure);

  // Search one round to get promising states
  PrintTitle("Search", verbose);
  best_states = SearchOneRound(num_random * 3, &random_states);

  // Infer bound. This is necessary for computing the correct ToStr() for redundancy check
  best_states = search_task->compute_dag.InferBound(best_states);
  random_states = search_task->compute_dag.InferBound(random_states);

  // Pick `num_measure_per_iter` states to measure, check hash to remove already measured state
  // Also pick some random states to do eps-greedy
  inputs = PickStatesWithEpsGreedy(best_states, random_states, num_measure);

  // Measure candidate states
  PrintTitle("Measure", verbose);
  results = measurer->Measure(search_task, GetRef<SearchPolicy>(this), inputs);

  // Update measured states throughputs. These states will join the EvolutionarySearch in later
  // search rounds.
  for (const auto& res : results) {
    measured_states_throughputs_.push_back(1.0 / FloatArrayMean(res->costs));
  }

  auto t_begin = std::chrono::high_resolution_clock::now();

  // Update the cost model
  PrintTitle("Train cost model", verbose);
  program_cost_model->Update(inputs, results);

  PrintTimeElapsed(t_begin, "training", verbose);

  return std::make_pair(std::move(inputs), std::move(results));
}

Array<State> SketchPolicyNode::SearchOneRound(int num_random_states, Array<State>* random_states) {
  // Get parameters
  int population = GetIntParam(params, SketchParamKey::EvolutionarySearch::population);
  int num_use_measured = std::min(
      static_cast<int>(measured_states_vector_.size()),
      static_cast<int>(
          GetDoubleParam(params, SketchParamKey::SampleInitPopulation::use_measured_ratio) *
          population));

  // 1. Generate sketches
  if (sketch_cache_.empty()) {
    sketch_cache_ = GenerateSketches();
  }

  // 2. Sample the init population
  Array<State> init_population = SampleInitPopulation(sketch_cache_);

  // 3. Perform evolutionary search.
  // Also insert already measured good states to the initial population
  std::vector<int> indices = Argsort(measured_states_throughputs_);
  for (int i = 0; i < num_use_measured; i++) {
    init_population.push_back(measured_states_vector_[indices[i]]);
  }
  // Sample some random states for eps-greedy
  if (num_random_states > 0 && random_states != nullptr) {
    *random_states = RandomSampleStates(init_population, &rand_gen, num_random_states);
  }
  return EvolutionarySearch(init_population, num_measure_per_iter_ * 2);
}

void generatePermutations(const std::unordered_map<int, Array<Array<Integer>>>& full_factor_list,
                          std::vector<Array<Integer>>& current_config, int depth,
                          std::map<int, ConfigKey>& conf_table, int& idx) {
  if (current_config.size() == depth) {
    ConfigKey config_key;
    for (auto& config_value : current_config) {
      for (auto& value : config_value) {
        config_key.push_back(static_cast<int>(value));
      }
    }
    conf_table[idx++] = config_key;
    return;
  }

  for (auto config_value : full_factor_list.at(current_config.size())) {
    if (config_value.size() == 1) {
      config_value.push_back(1);
    }
    current_config.push_back(config_value);
    generatePermutations(full_factor_list, current_config, depth, conf_table, idx);
    current_config.pop_back();
  }
}

ConfigKey SketchPolicyNode::map_to_configkey(std::unordered_map<std::string, std::vector<int>> current_config, std::vector<splitMeta*> v_splitMeta_info){
  State state = sketch_cache_[0];
  const State& init_state = this->search_task->compute_dag->init_state;
  std::map<int, int> stage_itr_offset;

  for (auto spm : v_splitMeta_info) {
    int step_id = spm->step_id;
    // std::cout << "v_splitMeta_info step_id " << step_id  << std::endl;
    auto ps = state->transform_steps[step_id].as<SplitStepNode>();
    int orgin_stage_id = ps->stage_id;
    auto ori_iters = (init_state)->stages[orgin_stage_id]->iters;
    // restore iterator id according to transform steps
    if (stage_itr_offset.find(orgin_stage_id) != stage_itr_offset.end()) {
      // accumulate the previous split
      int offset = stage_itr_offset[orgin_stage_id];
      spm->origin_itr = ori_iters[ps->iter_id - offset];

      
      stage_itr_offset[orgin_stage_id] += ps->lengths.size();
      if (ps->lengths.size() == 4) {
        spm->parallel = true;
      } else if (ps->lengths.size() == 2) {
        spm->parallel = false;
      } else {
        assert(false && "unknown itr type");
      }
    } else {
      // first one in the stage
      spm->origin_itr = ori_iters[ps->iter_id];
      // Fetch the current tile sizes.
      stage_itr_offset[orgin_stage_id] = ps->lengths.size();
      if (ps->lengths.size() == 4) {
        spm->parallel = true;
      } else if (ps->lengths.size() == 2) {
        spm->parallel = false;
      } else {
        assert(false && "unknown itr type");
      }
    }
  }

  // std::cout << "Done itr id adjust " << v_splitMeta_info.size() << std::endl;

  ConfigKey config_key;
  // std::cout << "v_splitMeta_info size: " << v_splitMeta_info.size() << std::endl;
  for (auto spm : v_splitMeta_info) {
    // std::cout << "spm : " << *spm << std::endl;
    if (spm->parallel == 1) {
      std::vector<int> tile_conf = current_config[spm->origin_itr->name];
      // std::cout << "tile_name " << spm->origin_itr->name << std::endl;
      for (int i = 0; i < tile_conf.size(); i++){
        // std::cout << tile_conf[i] << " ";
        config_key.push_back(tile_conf[i]);
      }
    } else {
      std::vector<int> tile_conf = current_config[spm->origin_itr->name];
      // std::cout << "tile_name " << spm->origin_itr->name << std::endl;
      for (int i = 0; i < tile_conf.size(); i++){
        // std::cout << tile_conf[i] << " ";
        config_key.push_back(tile_conf[i]);
      }
    }
    // if (spm == v_splitMeta_info.end()[-1]) {
    //   continue;
    // }
  }

  return config_key;
}

std::unordered_map<std::string, std::vector<int>> SketchPolicyNode::GetStateFactor(const SearchTask& task, const State& state){
  std::vector<splitMeta*> v_splitMeta_info;
  v_splitMeta_info = GenerateSplitMeta(this, state);
  State ret_state;
  StateNode* pstate;

  if (state->stages.empty()) {
    // If the input state is incomplete with empty operation stage
    // create a new state from init_state and update it first
    ret_state = task->compute_dag->init_state;
    pstate = ret_state.CopyOnWrite();
    pstate->transform_steps = state->transform_steps;
    for (const auto& step : pstate->transform_steps) {
      StepApplyToState(step, &ret_state, task->compute_dag);
    }
  } else {
    ret_state = state;
    pstate = ret_state.CopyOnWrite();
  }

  Array<te::Stage> stages;
  StageToAxesMap stage_to_axes;
  te::Schedule sch;
  Array<te::Tensor> tensors;
  // Replay steps to tvm::Schedule
  std::tie(sch, tensors) = task->compute_dag.ApplySteps(pstate->transform_steps, &stages, &stage_to_axes);
  sch = sch.normalize_for_feature_extraction();
  // Get bound information from TVM schedule
  Map<IterVar, Range> bounds = te::InferBound(sch);

  // Update the state bound information
  for (size_t i = 0; i < pstate->stages.size(); ++i) {
    const Stage& stage = pstate->stages[i];

    if (stage->compute_at == ComputeAtKind::kInlined) {
      continue;
    }

    Array<Iterator> new_iters;
    new_iters.reserve(stage->iters.size());
    // Get bound information from schedule
    // the StageToAxesMap is used to find the corresponding IterVar in TVM schedule result
    for (size_t j = 0; j < stage->iters.size(); ++j) {
      const Iterator& iter = stage->iters[j];
      const IterVar& axis = stage_to_axes.at(stages[i])[j];

      auto find_res = bounds.find(axis);
      if (find_res != bounds.end()) {
        new_iters.push_back(Iterator(iter->name, (*find_res).second, iter->iter_kind,
                                     iter->annotation, &iter->orig_iters));
      } else {
        LOG(FATAL) << "Infer bound fails";
      }
    }

    pstate->stages.Set(
        i, Stage(stage->op, stage->op_type, new_iters, stage->compute_at, stage->attrs));
  }
  const State& init_state = task->compute_dag->init_state;
  std::map<int, int> stage_itr_offset;

  for (auto spm : v_splitMeta_info) {
    int step_id = spm->step_id;
    // // std::cout << "v_splitMeta_info step_id " << step_id  << std::endl;
    auto ps = (state)->transform_steps[step_id].as<SplitStepNode>();
    int orgin_stage_id = ps->stage_id;
    auto ori_iters = (init_state)->stages[orgin_stage_id]->iters;
    // restore iterator id according to transform steps
    if (stage_itr_offset.find(orgin_stage_id) != stage_itr_offset.end()) {
      // accumulate the previous split
      int offset = stage_itr_offset[orgin_stage_id];
      spm->origin_itr = ori_iters[ps->iter_id - offset];

      // Fetch the current tile sizes.
      std::vector<int> lengths(ps->lengths.size() + 1, 1);
      for (int i = 0; i < static_cast<int>(ps->lengths.size()); ++i) {
        lengths[i + 1] = GetIntImm(ps->lengths[i].value());
      }
      lengths[0] = spm->problem_size / ElementProduct(lengths);
      for (auto elm : lengths) {
        spm->add_tilesize(elm);
      }
      stage_itr_offset[orgin_stage_id] += ps->lengths.size();
      if (lengths.size() == 5) {
        spm->parallel = true;
      } else if (lengths.size() == 3) {
        spm->parallel = false;
      } else {
        assert(false && "unknown itr type");
      }
    } else {
      // first one in the stage
      spm->origin_itr = ori_iters[ps->iter_id];
      // Fetch the current tile sizes.
      std::vector<int> lengths(ps->lengths.size() + 1, 1);
      for (int i = 0; i < static_cast<int>(ps->lengths.size()); ++i) {
        lengths[i + 1] = GetIntImm(ps->lengths[i].value());
      }
      lengths[0] = spm->problem_size / ElementProduct(lengths);
      for (auto elm : lengths) {
        spm->add_tilesize(elm);
      }
      stage_itr_offset[orgin_stage_id] = ps->lengths.size();
      if (lengths.size() == 5) {
        spm->parallel = true;
      } else if (lengths.size() == 3) {
        spm->parallel = false;
      } else {
        assert(false && "unknown itr type");
      }
    }
  }
  
  std::unordered_map<std::string, std::vector<int>> current_config;
  for (auto spm : v_splitMeta_info) {
    if (spm->parallel == 1) {
      std::vector<int> tile_conf;
      int reg = spm->tile_sizes[1] * spm->tile_sizes[3] * spm->tile_sizes[4];
      int tb = spm->tile_sizes[2];
      tile_conf.push_back(static_cast<int>(reg));
      tile_conf.push_back(static_cast<int>(tb));
      current_config[spm->origin_itr->name] = tile_conf;
    } else {
      int inner_outer = spm->tile_sizes[1];
      std::vector<int> tile_conf;
      tile_conf.push_back(static_cast<int>(inner_outer));
      tile_conf.push_back(1);
      current_config[spm->origin_itr->name] = tile_conf;
    }
    // if (spm == v_splitMeta_info.end()[-1]) {
    //   continue;
    // }
  }
  return current_config;
}

// UpDownMutate for current_config, using pz_factors
std::vector<ConfigKey> SketchPolicyNode::UpDownMutate(std::unordered_map<std::string, std::vector<int>> current_config, std::unordered_map<std::string, std::vector<int>> pz_factors, std::vector<splitMeta*> v_splitMeta_info){
  std::vector<ConfigKey> neighbors_config_key;

  //up-down mutate for tb or reg, push into the neighbors_config_key
  // int max_innermost_split_factor = GetIntParam(this->params, SketchParamKey::max_innermost_split_factor);
  // use INT_MAX to disable it temporarily
  int max_innermost_split_factor = INT_MAX;

  std::unordered_map<std::string, std::vector<int>> tmp_config = current_config;
  for (auto sm : v_splitMeta_info){
    // std::cout << "current_config: \n" << std::endl;
    for (auto c : tmp_config){
      std::string key = c.first;
      std::vector<int> value = c.second;
      // std::cout << "key: " << key << std::endl;
      for (int i = 0; i < value.size(); i++){
        // std::cout << value[i] << " ";
      }
      // std::cout << std::endl;
    }
    if (sm->parallel) { // mutate for this dimension and concrete tmp_config
      auto dim_name = sm->origin_itr->name;
      auto pz = sm->problem_size;
      auto reg = current_config[dim_name][0];
      auto tb = current_config[dim_name][1];

      // find the index of tb and reg in pz_factors
      auto tb_index = std::find(pz_factors[dim_name].begin(), pz_factors[dim_name].end(), tb);
      auto reg_index = std::find(pz_factors[dim_name].begin(), pz_factors[dim_name].end(), reg);

      // up for tb
      if (tb_index != pz_factors[dim_name].end() - 1){
        tmp_config = current_config;
        auto up_tb_index = tb_index + 1;
        // std::vector<int> tmp =pz_factors[dim_name];
        // auto up_tb = tmp.at(up_tb_index - pz_factors[dim_name].begin());
        auto up_tb = pz_factors[dim_name].at(up_tb_index - pz_factors[dim_name].begin());
        //valid
        if (up_tb * reg <= pz && up_tb <= max_innermost_split_factor){
          // std::cout << "up_tb: " << up_tb << std::endl;
          tmp_config[dim_name][1] = up_tb;
          ConfigKey config_key = map_to_configkey(tmp_config, v_splitMeta_info);
          neighbors_config_key.push_back(config_key);
        }
      }
      // down for tb
      if (tb_index != pz_factors[dim_name].begin()){
        tmp_config = current_config;
        auto down_tb_index = tb_index - 1;
        auto down_tb = pz_factors[dim_name].at(down_tb_index - pz_factors[dim_name].begin());
        //valid
        if (down_tb * reg <= pz && down_tb <= max_innermost_split_factor){
          // std::cout << "down_tb: " << down_tb << std::endl;
          tmp_config[dim_name][1] = down_tb;
          ConfigKey config_key = map_to_configkey(tmp_config, v_splitMeta_info);
          neighbors_config_key.push_back(config_key);
        }
      }

      // up for reg
      if (reg_index != pz_factors[dim_name].end() - 1){
        tmp_config = current_config;
        auto up_reg_index = reg_index + 1;
        auto up_reg = pz_factors[dim_name].at(up_reg_index - pz_factors[dim_name].begin());
        //valid
        if (up_reg * tb <= pz && up_reg <= max_innermost_split_factor){
          // std::cout << "up_reg: " << up_reg << "for dimname " << dim_name << std::endl;
          tmp_config[dim_name][0] = up_reg;
          ConfigKey config_key = map_to_configkey(tmp_config, v_splitMeta_info);
          neighbors_config_key.push_back(config_key);
        }
      }

      // down for reg
      if (reg_index != pz_factors[dim_name].begin()){
        tmp_config = current_config;
        auto down_reg_index = reg_index - 1;
        auto down_reg = pz_factors[dim_name].at(down_reg_index - pz_factors[dim_name].begin());
        //valid
        if (down_reg * tb <= pz && down_reg <= max_innermost_split_factor){
          // std::cout << "down_reg: " << down_reg << std::endl;
          tmp_config[dim_name][0] = down_reg;
          ConfigKey config_key = map_to_configkey(tmp_config, v_splitMeta_info);
          neighbors_config_key.push_back(config_key);
        }
      }
    }
    else{
      auto dim_name = sm->origin_itr->name;
      auto pz = sm->problem_size;
      auto inner_inner = current_config[dim_name][0];
      // std::cout << "inner_inner: " << inner_inner << std::endl;

      auto idx = std::find(pz_factors[dim_name].begin(), pz_factors[dim_name].end(), inner_inner);

      // up
      if (idx != pz_factors[dim_name].end() - 1){
        tmp_config = current_config;
        auto up_idx = idx + 1;
        auto up_inner_inner = pz_factors[dim_name].at(up_idx - pz_factors[dim_name].begin());
        //valid
        if (up_inner_inner <= pz && up_inner_inner <= max_innermost_split_factor){
          // std::cout << "up_inner_inner: " << up_inner_inner << "for dimname " << dim_name << std::endl;
          tmp_config[dim_name][0] = up_inner_inner;
          ConfigKey config_key = map_to_configkey(tmp_config, v_splitMeta_info);

          neighbors_config_key.push_back(config_key);
        }
      }
      // down
      if (idx != pz_factors[dim_name].begin()){
        tmp_config = current_config;
        auto down_idx = idx - 1;
        auto down_inner_inner = pz_factors[dim_name].at(down_idx - pz_factors[dim_name].begin());
        //valid
        if (down_inner_inner <= pz && down_inner_inner <= max_innermost_split_factor){
          // std::cout << "down_inner_inner: " << down_inner_inner << "for dimname " << dim_name << std::endl;
          tmp_config[dim_name][0] = down_inner_inner;
          ConfigKey config_key = map_to_configkey(tmp_config, v_splitMeta_info);
          neighbors_config_key.push_back(config_key);
        }
      }
    }
  }
  return neighbors_config_key;
}




std::string conf2string(const ConfigKey& conf) {
    std::string res = "";
    for (auto& c : conf) {
        res += std::to_string(c) + " ";
    }
    return res;
}

int containsConfigKey(const std::map<int, ConfigKey>& map, const ConfigKey& keyToFind) {
    int i = 0;
    for (const auto& pair : map) {
      if (pair.second == keyToFind) {
        return i;
      }
    }
    return 0;
}


std::map<int, ConfigKey> SketchPolicyNode::GenerateUniquetable(SketchPolicyNode* policy, State state, std::vector<splitMeta*> v_splitMeta_info, ConfigKey base, std::unordered_map<std::string, std::vector<int>> current_config){
  std::map<int, ConfigKey> res;

  const State& init_state = policy->search_task->compute_dag->init_state;
  std::map<int, int> stage_itr_offset;

  for (auto spm : v_splitMeta_info) {
    int step_id = spm->step_id;
    // std::cout << "v_splitMeta_info step_id " << step_id  << std::endl;
    auto ps = state->transform_steps[step_id].as<SplitStepNode>();
    int orgin_stage_id = ps->stage_id;
    auto ori_iters = (init_state)->stages[orgin_stage_id]->iters;
    // restore iterator id according to transform steps
    if (stage_itr_offset.find(orgin_stage_id) != stage_itr_offset.end()) {
      // accumulate the previous split
      int offset = stage_itr_offset[orgin_stage_id];
      spm->origin_itr = ori_iters[ps->iter_id - offset];

      
      stage_itr_offset[orgin_stage_id] += ps->lengths.size();
      if (ps->lengths.size() == 4) {
        spm->parallel = true;
      } else if (ps->lengths.size() == 2) {
        spm->parallel = false;
      } else {
        assert(false && "unknown itr type");
      }
    } else {
      // first one in the stage
      spm->origin_itr = ori_iters[ps->iter_id];
      // Fetch the current tile sizes.
      stage_itr_offset[orgin_stage_id] = ps->lengths.size();
      if (ps->lengths.size() == 4) {
        spm->parallel = true;
      } else if (ps->lengths.size() == 2) {
        spm->parallel = false;
      } else {
        assert(false && "unknown itr type");
      }
    }
  }

  // std::cout << "Done itr id adjust " << v_splitMeta_info.size() << std::endl;
  SplitFactorizationMemo sfm;
  int max_innermost_split_factor =
      GetIntParam(policy->params, SketchParamKey::max_innermost_split_factor);

  int tiling_level = 2;
  std::unordered_map<int, Array<Array<Integer>>> full_factor_list;
  int i = 0;
  for (auto sm : v_splitMeta_info){
    if (sm->parallel) {
      std::string dim_name = sm->origin_itr->name;
      // std::cout << "dim_name: " << dim_name << ", i = " << i << std::endl;
      auto fact_schem = sfm.GetFactorizationSchemes(sm->problem_size, tiling_level, max_innermost_split_factor);
      full_factor_list[i] = fact_schem;
      i++;
    } else {
      // non-parallel
      tiling_level = 1;
      std::string dim_name = sm->origin_itr->name;
      // std::cout << "dim_name: " << dim_name << ", i = " << i << std::endl;
      auto fact_schem = sfm.GetFactorizationSchemes(sm->problem_size, tiling_level, max_innermost_split_factor);
      full_factor_list[i] = fact_schem;
      i++;
    }
  }

  int idx = 0;
  int depth = i;
  std::vector<Array<Integer>> current;

  // permutate the base 


  //TODO: prune the config table
  generatePermutations(full_factor_list, current, depth, res, idx);

  return res;
}

/* Generate direct neighbors states for state
*  @param states: base states
*  @param pz_factors: factor list for problem size for example: dim_i = 6 --> factor[1, 2, 3, 6]
*  @return: neighbors states table
*/
std::vector<ConfigKey> SketchPolicyNode::GetDirectNeighbors(std::unordered_map<std::string, std::vector<int>> current_config, std::unordered_map<std::string, std::vector<int>> pz_factors, Array<State>& sketches, std::vector<splitMeta*> v_splitMeta_info){
  std::vector<ConfigKey> neighbors_config_key = UpDownMutate(current_config, pz_factors, v_splitMeta_info);
  return neighbors_config_key;
}

std::unordered_map<std::string, std::vector<int>> ConfigKey2Map(const ConfigKey& conf, std::vector<splitMeta*> v_splitMeta_info) {
    std::unordered_map<std::string, std::vector<int>> res;
    int idx = 0;
    for (auto& sm : v_splitMeta_info) {
        std::vector<int> tmp;
        if (sm->parallel) {
            tmp.push_back(conf[idx]);
            tmp.push_back(conf[idx + 1]);
            idx += 2;
        } else {
            tmp.push_back(conf[idx]);
            tmp.push_back(1);
            idx += 2;
        }
        res[sm->origin_itr->name] = tmp;
    }
    // std::cout << "ConfigKey2Map: " << std::endl;
    // for (auto c: conf) {
    //     std::cout << c << " ";
    // }
    // for (auto& c : res) {
    //     std::cout << c.first << ": ";
    //     for (auto& v : c.second) {
    //         std::cout << v << " ";
    //     }
    //     std::cout << std::endl;
    // }
    return res;
}


std::string ConfigKey2string(const ConfigKey& conf) {
    std::string res = "";
    for (auto& c : conf) {
        res += std::to_string(c) + "_";
    }
    // std::cout << "ConfigKey2string: " << res << std::endl;
    return res;
}

/* Generate neighbors states for all base states
*  @param states: base states
*  @param pz_factors: factor list for problem size for example: dim_i = 6 --> factor[1, 2, 3, 6]
*  @return: neighbors states table
*/
Array<Array<State>> SketchPolicyNode::GenerateNeighbours(Array<State> states, std::unordered_map<std::string, std::vector<int>> pz_factors, Array<State>& sketches, std::vector<splitMeta*> v_splitMeta_info){
  Array<Array<State>> neighbour_table;
  std::vector<std::vector<ConfigKey>> all_neighbors_config_key; 
  for (auto& state_ite : states) {
    State state; // current base
    state = state_ite;
    std::unordered_set<std::string> neighbors_remove_dup;
    std::unordered_map<std::string, std::vector<int>> current_base = GetStateFactor(search_task, state);
    std::vector<ConfigKey> neighbors_conf_key; //store all the neighbors' config key

    // avoid base state been added to neighbors
    std::vector<splitMeta*> base_meta_info = GenerateSplitMeta(this, state);
    neighbors_remove_dup.insert(state_to_string(state, base_meta_info, search_task));

    // get direct neighbors
    std::unordered_map<std::string, std::vector<int>> current_config = GetStateFactor(search_task, state);
    std::vector<ConfigKey> direct_neighbors_config_key = GetDirectNeighbors(current_config, pz_factors, sketches, v_splitMeta_info);

    for (auto n : direct_neighbors_config_key){
      const auto state_str = ConfigKey2string(n);
      if (neighbors_remove_dup.count(state_str) == 0) {
        neighbors_remove_dup.insert(state_str);
        neighbors_conf_key.push_back(n);
      }
    }

    // get diagnal neighbors
    for (auto n : direct_neighbors_config_key){
      std::unordered_map<std::string, std::vector<int>> current_map = ConfigKey2Map(n, v_splitMeta_info);
      std::vector<ConfigKey> tmp = GetDirectNeighbors(current_map, pz_factors, sketches, v_splitMeta_info);
      for (auto t : tmp){
        const auto state_str = ConfigKey2string(t);
        if (neighbors_remove_dup.count(state_str) == 0) {
          neighbors_remove_dup.insert(state_str);
          neighbors_conf_key.push_back(t);
        }
      }
    }

    all_neighbors_config_key.push_back(neighbors_conf_key);
  }

  int idx = 0;
  for (auto& state_ite : states) {
    State state; // current base
    state = state_ite;
    
    std::map<int, ConfigKey> tmp_conf_table;
    std::vector<ConfigKey> tmp_neighbors_conf_key = all_neighbors_config_key[idx];
    int i = 0;
    for (auto& n : tmp_neighbors_conf_key) {
      tmp_conf_table[i++] = n;
    }
    Array<State> sampled_states = SampleUniquePopulation(tmp_conf_table, sketches, v_splitMeta_info);
    Array<State> neighbors;
    neighbors.push_back(state);
    for (auto& s : sampled_states) {
      neighbors.push_back(s);
    }
    neighbour_table.push_back(neighbors);
    idx += 1;
  }

  return neighbour_table;
}



/* decide move
*  @param neighbour_table: base state with its neighbours
*  @param next: next states
*  @return: local mins and next states
*/
Array<State> SketchPolicyNode::NodeMove(Array<Array<State>> neighbour_table, Array<State>* next_states) {
    auto tic_begin = std::chrono::high_resolution_clock::now();

    // Clear next_states
    next_states->clear();

    Array<State> local_min;
    std::mutex next_states_mutex, visited_mutex, local_min_mutex;

    //calculate the pop_scores for every path, later accecible by index
    std::vector<std::vector<float>> vec_pop_scores;
    for (auto path : neighbour_table){
      if (path.empty()) {
          std::cout << "path is empty" << std::endl;
          continue;
      }
      std::vector<float> tmp;
      tmp.reserve(path.size());
      program_cost_model->Predict(search_task, path, &tmp);
      vec_pop_scores.push_back(tmp);
    }

    support::parallel_for(0, neighbour_table.size(), [this, &neighbour_table, &vec_pop_scores, next_states, &local_min, &next_states_mutex, &visited_mutex, &local_min_mutex](int index) {
      const auto local_path = neighbour_table[index];

      if (local_path.empty()) {
        std::cout << "path is empty" << std::endl;
        return;
      }

      std::vector<float> pop_scores = vec_pop_scores[index];

      if (pop_scores.size() - 1 == 0 || pop_scores[0] == -std::numeric_limits<float>::infinity()) {
        // Invalid and no neighbors
        return;
      }

      // TODO(Chendi): sorted by score
      // Determine if the neighbor should be a local minimum
      // path[0] : base state
      // path[1:] : neighbour states
      float base_score = pop_scores[0];
      float best_score = base_score;
      int best_neighbour_index = 0;
      bool real_local_min = true;

      for (int i = 1; i < pop_scores.size(); i++) {
        std::vector<splitMeta*> v_splitMeta_info = GenerateSplitMeta(this, local_path[i]);
        const auto state_str = state_to_string(local_path[i], v_splitMeta_info, search_task);

        // check if it is a real local min, all neighbor pscores should be not better than base
        // pscore
        if (pop_scores[i] > -1e10 && pop_scores[i] > best_score) {
          real_local_min = false;
        }
        // find a better one, update best_score
        if (pop_scores[i] > -1e10 && pop_scores[i] > best_score &&
            visited.count(state_str) == 0) {
          best_neighbour_index = i;
          best_score = pop_scores[i];
        }
      }

      // local min or not
      if (best_neighbour_index == 0) {
        if (real_local_min) {  // no absolute better pscore neighbor, real local min
          {
            std::lock_guard<std::mutex> lock(local_min_mutex);
            local_min.push_back(local_path[0]);  // send out local_min to measure
          }

          // may have equal pscore neighbors
          // in case we miss any near boundary states, equal best pscore states should also be
          // explored
          for (int i = 0; i < pop_scores.size(); i++) {
            if (i == 0) continue;  // skip base state

            std::vector<splitMeta*> v_splitMeta_info = GenerateSplitMeta(this, local_path[i]);
            const auto state_str =
                state_to_string(local_path[i], v_splitMeta_info, search_task);

            // add unvisited equal pscore states to next_states
            if (pop_scores[i] > -1e10 && pop_scores[i] == pop_scores[0] &&
                visited.count(state_str) == 0) {
              std::lock_guard<std::mutex> lock_visited(visited_mutex);
              {
                visited.insert(state_str);  // add all next_states to visited, avoid circular
              }
              std::lock_guard<std::mutex> lock(next_states_mutex);
              { next_states->push_back(local_path[i]); }
            }
          }
        }
      } else {
        // General case, better neighbour to move
        std::vector<splitMeta*> v_splitMeta_info =
            GenerateSplitMeta(this, local_path[best_neighbour_index]);
        const auto state_str =
            state_to_string(local_path[best_neighbour_index], v_splitMeta_info, search_task);
        std::lock_guard<std::mutex> lock_visited(visited_mutex);
        {
          // if best_neighbour_index != 0, it has never been visited, no need to check visited,
          // direct insert
          visited.insert(state_str);  // add all next_states to visited, avoid circular
        }

        std::lock_guard<std::mutex> lock(next_states_mutex);
        {
          next_states->push_back(
              local_path[best_neighbour_index]);  // move to better predict neighbour
        }
      }
    });

    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - tic_begin).count();
    std::cout << "NodeMove Time elapsed: " << std::fixed << std::setprecision(5) << duration << std::endl;
    // std::cout << "[NodeMove] Number of next_states : " << next_states->size() << std::endl;

    // TO measure local min
    return local_min;
}

std::unordered_map<std::string, std::vector<int>>  SketchPolicyNode::GetFactorInfo(SketchPolicyNode* policy, State* state, std::vector<splitMeta*> v_splitMeta_info){
  std::unordered_map<std::string, std::vector<int>> res;
  const State& init_state = policy->search_task->compute_dag->init_state;
  std::map<int, int> stage_itr_offset;

  for (auto spm : v_splitMeta_info) {
    int step_id = spm->step_id;
    // std::cout << "v_splitMeta_info step_id " << step_id  << std::endl;
    auto ps = (*state)->transform_steps[step_id].as<SplitStepNode>();
    int orgin_stage_id = ps->stage_id;
    auto ori_iters = (init_state)->stages[orgin_stage_id]->iters;
    // restore iterator id according to transform steps
    if (stage_itr_offset.find(orgin_stage_id) != stage_itr_offset.end()) {
      // accumulate the previous split
      int offset = stage_itr_offset[orgin_stage_id];
      spm->origin_itr = ori_iters[ps->iter_id - offset];

      
      stage_itr_offset[orgin_stage_id] += ps->lengths.size();
      if (ps->lengths.size() == 4) {
        spm->parallel = true;
      } else if (ps->lengths.size() == 2) {
        spm->parallel = false;
      } else {
        assert(false && "unknown itr type");
      }
    } else {
      // first one in the stage
      spm->origin_itr = ori_iters[ps->iter_id];
      // Fetch the current tile sizes.
      stage_itr_offset[orgin_stage_id] = ps->lengths.size();
      if (ps->lengths.size() == 4) {
        spm->parallel = true;
      } else if (ps->lengths.size() == 2) {
        spm->parallel = false;
      } else {
        assert(false && "unknown itr type");
      }
    }
  }

  // std::cout << "Done itr id adjust " << v_splitMeta_info.size() << std::endl;
  SplitFactorizationMemo sfm;
  int max_innermost_split_factor =
      GetIntParam(policy->params, SketchParamKey::max_innermost_split_factor);


  // get factor list for each dimension using GetFactorizationSchemes
  
  // get factor list for each dimension
  for (auto sm : v_splitMeta_info){
    if (sm->parallel) {
      auto dim_name = sm->origin_itr->name;
      // auto fact_schem = sfm.GetFactorizationSchemes(sm->problem_size, 2, max_innermost_split_factor);
      auto fact_schem = sfm.GetFactors(sm->problem_size);
      // std::cout << "fact_schem size " << fact_schem.size() << std::endl;
      // for (auto f : fact_schem){
      //   std::cout << f << " ";
      // }
      std::vector<int> v_fact_schem;
      for (auto f : fact_schem){
        v_fact_schem.push_back(f);
      }
      res[dim_name] = v_fact_schem;
    }
    else{
      auto dim_name = sm->origin_itr->name;
      // auto fact_schem = sfm.GetFactorizationSchemes(sm->problem_size, 1, max_innermost_split_factor);
      auto fact_schem = sfm.GetFactors(sm->problem_size);
      // std::cout << "fact_schem size " << fact_schem.size() << std::endl;
      // for (auto f : fact_schem){
      //   std::cout << f << " ";
      // }
      std::vector<int> v_fact_schem;
      for (auto f : fact_schem){
        v_fact_schem.push_back(f);
      }
      res[dim_name] = v_fact_schem;
    }
  }

  return res;
}

Array<State> SketchPolicyNode::SearchOneRoundPruePredict(int num_random_states, Array<State>* next_states, bool firsttime_random) {
  // std::cout << "[SearchOneRoundPruePredict] next_states size: " << next_states->size() << std::endl;
  // PrintTitle("Search", verbose);
  // 1. Generate sketches
  if (sketch_cache_.empty()) {
    sketch_cache_ = GenerateSketches();
    assert(sketch_cache_.size() == 1);
  }
  State state = sketch_cache_[0];
  std::vector<splitMeta*> v_splitMeta_info;
  v_splitMeta_info = GenerateSplitMeta(this, state);

  // TODO:(Chendi) can move to global variable
  std::unordered_map<std::string, std::vector<int>> pz_factors;
  pz_factors = GetFactorInfo(this, &state, v_splitMeta_info); // Calculate factor list problem size --> 6 factor[1, 2, 3, 6]
  
  // PrintTitle("Generate Base States", verbose);
  // base states in the init population
  Array<State> init_population;
  if (firsttime_random){
    // Sample the init population
    auto tic_begin = std::chrono::high_resolution_clock::now();
    init_population = SampleCUDAPopulation(sketch_cache_, num_random_states);
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                            std::chrono::high_resolution_clock::now() - tic_begin)
                            .count();
    StdCout(verbose) << "SampleCUDAPopulation Time elapsed: " << std::fixed
                      << std::setprecision(5) << duration << std::endl;

    auto tic_begin1 = std::chrono::high_resolution_clock::now();
    // push num_random_states random states into init_population as start point
    Array<State> base_population;
    for (auto state : init_population){
      std::vector<splitMeta*> tmp_meta_info = GenerateSplitMeta(this, state);
      const auto state_str = state_to_string(state, tmp_meta_info, search_task);
      if(visited.count(state_str) == 0) {
        visited.insert(state_str);
        base_population.push_back(state);
        if (--num_random_states == 0){
          break;
        }
      }
    }
    double duration1 = std::chrono::duration_cast<std::chrono::duration<double>>(
                            std::chrono::high_resolution_clock::now() - tic_begin1)
                            .count();
    StdCout(verbose) << "GenerateSplitMeta Time elapsed: " << std::fixed
                      << std::setprecision(5) << duration1 << std::endl;
    init_population = base_population;
  }
  else{
    init_population.clear();
    for (auto s : *next_states){
      init_population.push_back(s);
    }
  }

  std::cout << "Base nodes #" << init_population.size() << std::endl;
  PrintTitle("Generate Neighbours", verbose);
  auto tic_begin2 = std::chrono::high_resolution_clock::now();

  Array<Array<State>> neighbour_table = GenerateNeighbours(init_population, pz_factors, sketch_cache_, v_splitMeta_info); 
  
  double duration2 = std::chrono::duration_cast<std::chrono::duration<double>>(
                            std::chrono::high_resolution_clock::now() - tic_begin2)
                            .count();
  StdCout(verbose) << "GenerateNeighbours Time elapsed: " << std::fixed
                      << std::setprecision(5) << duration2 << std::endl;
  PrintTitle("Node Move", verbose);

  return NodeMove(neighbour_table, next_states);
}


Array<State> SketchPolicyNode::GenerateSketches() {
  const State& init_state = search_task->compute_dag->init_state;

  // Two ping pong buffers to avoid copy
  Array<State> states_buf1{init_state}, states_buf2;
  Array<State>* pnow = &states_buf1;
  Array<State>* pnext = &states_buf2;

  // A map that maps state to its current working position (stage_id)
  std::unordered_map<State, int, ObjectHash, ObjectEqual> cur_stage_id_map;
  cur_stage_id_map[init_state] = static_cast<int>(init_state->stages.size()) - 1;

  // Derivation rule based enumeration
  Array<State> out_states;
  while (!pnow->empty()) {
    pnext->clear();
    for (const State& state : *pnow) {
      int stage_id = cur_stage_id_map[state];

      // Reaches to the terminal stage
      if (stage_id < 0) {
        out_states.push_back(state);
        continue;
      }

      // Try all derivation rules
      for (const auto& rule : sketch_rules) {
        auto cond = rule->MeetCondition(*this, state, stage_id);
        if (cond != SketchGenerationRule::ConditionKind::kSkip) {
          for (const auto& pair : rule->Apply(*this, state, stage_id)) {
            cur_stage_id_map[pair.first] = pair.second;
            pnext->push_back(pair.first);
          }
          // Skip the rest rules
          if (cond == SketchGenerationRule::ConditionKind::kApplyAndSkipRest) {
            break;
          }
        }
      }
    }
    std::swap(pnow, pnext);
  }

  // Hack for rfactor: Replace the split factor for rfactor to the undefined Expr(),
  // so later we can sample random value for the split factor.
  // Why don't we use Expr() when doing the split for rfactor at the first time?
  // Because during ApplySteps, a rfactor with undefined Expr() will crash TVM.
  // So rfactor with undefined Expr() will conflict with cache_write, cache_read, rfactor
  // in other stages
  for (size_t i = 0; i < out_states.size(); ++i) {
    auto state = out_states[i];
    auto pstate = state.CopyOnWrite();
    for (size_t step_id = 0; step_id < pstate->transform_steps.size(); ++step_id) {
      if (pstate->transform_steps[step_id]->IsInstance<RfactorStepNode>()) {
        ICHECK_GE(step_id, 1);
        int split_step_id = static_cast<int>(step_id - 1);
        auto step = pstate->transform_steps[split_step_id].as<SplitStepNode>();
        ICHECK(step != nullptr);
        pstate->transform_steps.Set(
            split_step_id, SplitStep(step->stage_id, step->iter_id, step->extent, {NullOpt},
                                     step->inner_to_outer));
      }
    }
    out_states.Set(i, std::move(state));
  }

  StdCout(verbose) << "Generate Sketches\t\t#s: " << out_states.size() << std::endl;
  return out_states;
}

Array<State> SketchPolicyNode::SampleUniquePopulation(std::map<int, ConfigKey> conf_table, Array<State>& sketches, std::vector<splitMeta*> v_splitMeta_info) {
  // Use this population as the parallel degree to do sampling
  
  int population = conf_table.size();

  assert(sketches.size() == 1); 

  int fail_ct = 0;
  Array<State> out_states;
  std::vector<std::mt19937> rand_gens;
  rand_gens.reserve(population);
  for (int i = 0; i < population; i++) {
    rand_gens.push_back(std::mt19937(rand_gen()));
  }

  std::unordered_set<std::string> explored_state_strs;
  // size_t iter = 1;
  size_t unchange_cnt = 0;
  
  std::vector<State> temp_states(population);

  std::vector<int> split_id;
  //std::cout<< "SampleUniquePopulation function---> uniq pop" << population << std::endl;
  for (auto sm : v_splitMeta_info){
    //std::cout << *sm << std::endl;
    split_id.push_back(sm->step_id);
  }
  support::parallel_for(0, population, [this, &temp_states, &sketches, &rand_gens, &conf_table, &split_id](int index) {
    // Apply random annotation rules one by one
    bool valid = true;
    InitFillTileSizeUnique cust_rule;
    InitUnroll cust_rule1;
    InitThreadBind cust_rule2;
    std::vector<PopulationGenerationRule*> cust_init_rules;
    cust_init_rules.push_back(&cust_rule2);
    cust_init_rules.push_back(&cust_rule1);


    ConfigKey tile_config = conf_table[index];

    State tmp_s = sketches[0];  // TODO: make muiltple sketch work later
    
    // std::cout << "before Apply_unique" << std::endl;
    if (cust_rule.Apply_unique(this, &tmp_s, tile_config, split_id) ==  
      PopulationGenerationRule::ResultKind::kInvalid) {
      valid = false;
    }
    // std::cout << "Done Apply_unique , valid: " << valid << std::endl;

    for (const auto& rule : cust_init_rules) {
      if (rule->Apply(this, &tmp_s, &rand_gens[index]) ==
          PopulationGenerationRule::ResultKind::kInvalid) {
        valid = false;
        break;
      }
    }
    // std::cout << "done apply cust_init_rules, valid: " << valid << std::endl;
    if (valid) {
      // std::cout << "success: state move to temp_states" << std::endl;
      // std::cout << tmp_s << std::endl;

      temp_states[index] = std::move(tmp_s);
    }
  }); // parallel generate
  
  // std::cout << "Done parallel generate" << std::endl;
  // Filter out the states that were failed to apply initial rules
  Array<State> cand_states;
  int i = 0;
  for (auto tmp_s : temp_states) {
    if (tmp_s.defined()) {
      // std:: cout << "index: " << i << " tmp_s is defined" << std::endl;
      cand_states.push_back(std::move(tmp_s));
    } else {
      // std::cout << "index: " << i << " tmp_s is not defined" << std::endl;
      fail_ct++;
    }
    i++;
  }
  // std::cout << "before pruning invalid state, cand_states size: " << cand_states.size() << std::endl;

  unchange_cnt++;
  if (!cand_states.empty()) {
    // std::vector<float> pop_scores;
    // pop_scores.reserve(cand_states.size());
    // cand_states = search_task->compute_dag.InferBound(cand_states);
    // PruneInvalidState(search_task, &cand_states);
    // program_cost_model->Predict(search_task, cand_states, &pop_scores);

    // std::cout << "compute dag: infer bound" << std::endl;
    cand_states = search_task->compute_dag.InferBound(cand_states);
    
    // std::cout << "pruning the invalid state" << std::endl;
    PruneInvalidState(search_task, &cand_states);
    // std::cout << "after pruning invalid state, cand_states size: " << cand_states.size() << std::endl;
    // TODO: check duplicate if generated code is same
    for (size_t i = 0; i < cand_states.size(); i++) {
      out_states.push_back(std::move(cand_states[i]));
    }
  }
  // std::cout << "after pruning, out_states size: " << out_states.size() << std::endl;
  // std::cout << "fail_ct: " << fail_ct << std::endl;cuda_view

  return out_states;
}

// we prefer sample from cuda view, so we need to check if the current config is cuda view prefer
bool SketchPolicyNode::cuda_view(const State& state, std::unordered_map<std::string, std::vector<int>> current_config, std::vector<splitMeta*> v_splitMeta_info){

  ConfigKey config_key;
  // std::cout << "size of v_splitMeta_info: " << v_splitMeta_info.size() << std::endl;
  int total_tb_size = 1;
  int totalReg = 1;
  for (auto spm : v_splitMeta_info) {
    // std::cout << "spm : " << *spm << std::endl;
    if (spm->parallel == 1) {// filter tb size for parallel dims
      int reg = spm->tile_sizes[1] * spm->tile_sizes[3] * spm->tile_sizes[4];
      int tb = spm->tile_sizes[2];
      int grid = spm->tile_sizes[0];
      totalReg *= reg;
      total_tb_size *= tb;
      // (Chendi)further prune?
    } else {// and filter inner_outer for reduce dims
      int outer_SM = spm->tile_sizes[0];
      int inner_outer = spm->tile_sizes[1];
      int inner_inner = spm->tile_sizes[2];
      // (Chendi)further prune?
    }
  }
  if (totalReg > 256){
    return false;
  }
  if (total_tb_size > 1024 || total_tb_size < 32){
    return false;
  }
  return true;
}

Array<State> SketchPolicyNode::SampleCUDAPopulation(const Array<State>& sketches, int num_required) {
  PrintTitle("Sample CUDA View Population", verbose);
  // Use this population as the parallel degree to do sampling
  int population = num_required*2;
  sample_init_min_pop_ = num_required;
  auto tic_begin = std::chrono::high_resolution_clock::now();

  int fail_ct = 0;
  Array<State> out_states;
  std::vector<std::mt19937> rand_gens;
  rand_gens.reserve(population);
  for (int i = 0; i < population; i++) {
    rand_gens.push_back(std::mt19937(rand_gen()));
  }

  std::unordered_set<std::string> explored_state_strs;
  size_t iter = 1;
  size_t unchange_cnt = 0;
  while (static_cast<int>(out_states.size()) < sample_init_min_pop_) {
    std::vector<State> temp_states(population);

    // Sample a batch of states randomly
    // TODO(Chendi): apply capacity prune here
    support::parallel_for(0, population, [this, &temp_states, &sketches, &rand_gens](int index) {
      // Randomly choose a sketch
      State tmp_s = sketches[(rand_gens[index])() % sketches.size()];
      // Apply random annotation rules one by one
      bool valid = true;
      for (const auto& rule : init_rules) {
        if (rule->Apply(this, &tmp_s, &rand_gens[index]) ==
            PopulationGenerationRule::ResultKind::kInvalid) {
          valid = false;
          break;
        }
      }
      if (valid) {
        temp_states[index] = std::move(tmp_s);
      }
    });

    // Filter out the states that were failed to apply initial rules
    Array<State> cand_states;
    for (auto tmp_s : temp_states) {
      if (tmp_s.defined()) {
        cand_states.push_back(std::move(tmp_s));
      } else {
        fail_ct++;
      }
    }

    unchange_cnt++;
    if (!cand_states.empty()) {
      // Run the cost model to make filter out states that failed to extract features.
      // This may happen due to illegal schedules or the schedules that uses too much
      // memory on GPU.
      cand_states = search_task->compute_dag.InferBound(cand_states);
      PruneInvalidState(search_task, &cand_states);

      for (size_t i = 0; i < cand_states.size(); i++) {
        // skip cache_failed
        if (cache_failed.count(cand_states[i].ToStr()) != 0) {
          continue;
        }
        // failure visited use toStr() to avoid
        std::vector<splitMeta*> v_splitMeta_info;
        v_splitMeta_info = GenerateSplitMeta(this, cand_states[i]);
        const auto state_str = state_to_string(cand_states[i], v_splitMeta_info, search_task);
        std::unordered_map<std::string, std::vector<int>> current_config = GetStateFactor(search_task, cand_states[i]);
        bool isCudaView = cuda_view(cand_states[i], current_config, v_splitMeta_info);
        if (isCudaView && explored_state_strs.count(state_str) == 0 && visited.count(state_str) == 0) {
          explored_state_strs.insert(state_str);
          out_states.push_back(std::move(cand_states[i]));
          unchange_cnt = 0;  // Reset the counter once we found a valid state
        } else if (!isCudaView) { // count cuda view failed, bring pop/2 to here
          fail_ct++;
          // cache all sampled population
          cache_failed.insert(cand_states[i].ToStr());
        }
      }
    }

    if (iter % 50 == 0) {
      double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                            std::chrono::high_resolution_clock::now() - tic_begin)
                            .count();
      StdCout(verbose) << "Sample Iter: " << iter << std::fixed << std::setprecision(4)
                       << "\t#Pop: " << out_states.size() << "\t#Target: " << sample_init_min_pop_
                       << "\tCUDA view fail_ct: " << fail_ct << "\tTime elapsed: " << std::fixed
                       << std::setprecision(2) << duration << std::endl;
    }

    if (unchange_cnt == 5) {
      // Reduce the target size to avoid too-long time in this phase if no valid state was found
      // in the past iterations
      if (sample_init_min_pop_ > 1) {
        sample_init_min_pop_ /= 2;
        StdCout(verbose) << "#Target has been reduced to " << sample_init_min_pop_
                         << " due to too many failures or duplications" << std::endl;
      }
      unchange_cnt = 0;
    }
    iter++;
  }

  double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - tic_begin)
                        .count();
  StdCout(verbose) << "Sample \t#s: " << out_states.size()
                   << "\tCUDA view fail_ct: " << fail_ct << "\tTime elapsed: " << std::fixed
                   << std::setprecision(2) << duration << std::endl;
  return out_states;
}

Array<State> SketchPolicyNode::SampleInitPopulation(const Array<State>& sketches) {
  PrintTitle("Sample Initial Population", verbose);
  // Use this population as the parallel degree to do sampling
  int population = GetIntParam(params, SketchParamKey::EvolutionarySearch::population);
  auto tic_begin = std::chrono::high_resolution_clock::now();

  int fail_ct = 0;
  Array<State> out_states;
  std::vector<std::mt19937> rand_gens;
  rand_gens.reserve(population);
  for (int i = 0; i < population; i++) {
    rand_gens.push_back(std::mt19937(rand_gen()));
  }

  std::unordered_set<std::string> explored_state_strs;
  size_t iter = 1;
  size_t unchange_cnt = 0;
  while (static_cast<int>(out_states.size()) < sample_init_min_pop_) {
    std::vector<State> temp_states(population);

    // Sample a batch of states randomly
    // TODO(Chendi): apply capacity prune here
    support::parallel_for(0, population, [this, &temp_states, &sketches, &rand_gens](int index) {
      // Randomly choose a sketch
      State tmp_s = sketches[(rand_gens[index])() % sketches.size()];
      // Apply random annotation rules one by one
      bool valid = true;
      for (const auto& rule : init_rules) {
        if (rule->Apply(this, &tmp_s, &rand_gens[index]) ==
            PopulationGenerationRule::ResultKind::kInvalid) {
          valid = false;
          break;
        }
      }
      if (valid) {
        temp_states[index] = std::move(tmp_s);
      }
    });

    // Filter out the states that were failed to apply initial rules
    Array<State> cand_states;
    for (auto tmp_s : temp_states) {
      if (tmp_s.defined()) {
        cand_states.push_back(std::move(tmp_s));
      } else {
        fail_ct++;
      }
    }

    unchange_cnt++;
    if (!cand_states.empty()) {
      // Run the cost model to make filter out states that failed to extract features.
      // This may happen due to illegal schedules or the schedules that uses too much
      // memory on GPU.
      std::vector<float> pop_scores;
      pop_scores.reserve(cand_states.size());
      cand_states = search_task->compute_dag.InferBound(cand_states);
      PruneInvalidState(search_task, &cand_states);
      program_cost_model->Predict(search_task, cand_states, &pop_scores);

      for (size_t i = 0; i < cand_states.size(); i++) {
        std::vector<splitMeta*> v_splitMeta_info;
        v_splitMeta_info = GenerateSplitMeta(this, cand_states[i]);
        const auto state_str = state_to_string(cand_states[i], v_splitMeta_info, search_task);
        if (pop_scores[i] > -1e10 && explored_state_strs.count(state_str) == 0 && visited.count(state_str) == 0) {
          explored_state_strs.insert(state_str);
          out_states.push_back(std::move(cand_states[i]));
          unchange_cnt = 0;  // Reset the counter once we found a valid state
        } else {
          fail_ct++;
        }
      }
    }

    if (iter % 5 == 0) {
      double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                            std::chrono::high_resolution_clock::now() - tic_begin)
                            .count();
      StdCout(verbose) << "Sample Iter: " << iter << std::fixed << std::setprecision(4)
                       << "\t#Pop: " << out_states.size() << "\t#Target: " << sample_init_min_pop_
                       << "\tfail_ct: " << fail_ct << "\tTime elapsed: " << std::fixed
                       << std::setprecision(2) << duration << std::endl;
    }

    if (unchange_cnt == 5) {
      // Reduce the target size to avoid too-long time in this phase if no valid state was found
      // in the past iterations
      if (sample_init_min_pop_ > 1) {
        sample_init_min_pop_ /= 2;
        StdCout(verbose) << "#Target has been reduced to " << sample_init_min_pop_
                         << " due to too many failures or duplications" << std::endl;
      }
      unchange_cnt = 0;
    }
    iter++;
  }

  double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - tic_begin)
                        .count();
  StdCout(verbose) << "Sample Initial Population\t#s: " << out_states.size()
                   << "\tfail_ct: " << fail_ct << "\tTime elapsed: " << std::fixed
                   << std::setprecision(2) << duration << std::endl;
  return out_states;
}

Array<State> SketchPolicyNode::EvolutionarySearch(const Array<State>& init_population,
                                                  int out_size) {
  Array<State> best_states;
  auto tic_begin = std::chrono::high_resolution_clock::now();

  size_t population = GetIntParam(params, SketchParamKey::EvolutionarySearch::population);
  double mutation_prob = GetDoubleParam(params, SketchParamKey::EvolutionarySearch::mutation_prob);
  int num_iters = GetIntParam(params, SketchParamKey::EvolutionarySearch::num_iters);

  // bool is_cost_model_reasonable = !program_cost_model->IsInstance<RandomModelNode>();
  // if (!is_cost_model_reasonable && num_iters > 2) {
  //   num_iters = 2;
  //   StdCout(verbose) << "GA iteration number has been adjusted to " << num_iters
  //                    << " due to random cost model" << std::endl;
  // }

  if (program_cost_model->IsInstance<PythonBasedModelNode>()){
    std::cout<< "PythonBasedModelNode " << std::endl;
  }
  else if (program_cost_model->IsInstance<RandomModelNode>()){
    std::cout<< "RandomModelNode " << std::endl;
  }
    else if (program_cost_model->IsInstance<AnaModelNode>()){
    std::cout<< "AnaModelNode " << std::endl;
  }
  std::cout<< "EvolutionarySearch num_iters " << num_iters<< std::endl;
  
  // Two ping pong buffers to avoid copy.
  Array<State> states_buf1{init_population}, states_buf2;
  states_buf1.reserve(population);
  states_buf2.reserve(population);
  Array<State>* pnow = &states_buf1;
  Array<State>* pnext = &states_buf2;

  // A heap to keep the best states during evolution
  using StateHeapItem = std::pair<State, float>;
  auto cmp = [](const StateHeapItem& left, const StateHeapItem& right) {
    return left.second > right.second;
  };
  std::vector<StateHeapItem> heap;
  std::unordered_set<std::string> in_heap(measured_states_set_);
  heap.reserve(out_size);

  // auxiliary global variables
  std::vector<float> pop_scores;
  std::vector<double> pop_selection_probs;
  float max_score = -1e-10f;
  pop_scores.reserve(population);
  pop_selection_probs.reserve(population);
  std::uniform_real_distribution<> dis(0.0, 1.0);

  // mutation rules
  int mutation_success_ct, mutation_fail_ct;
  mutation_success_ct = mutation_fail_ct = 0;
  std::vector<float> rule_weights;
  std::vector<double> rule_selection_probs;
  for (const auto& rule : mutation_rules) {
    rule_weights.push_back(rule->weight);
  }
  ComputePrefixSumProb(rule_weights, &rule_selection_probs);

  // Genetic Algorithm
  for (int k = 0; k < num_iters + 1; ++k) {
    // Maintain the heap
    *pnow = search_task->compute_dag.InferBound(*pnow);
    PruneInvalidState(search_task, pnow);
    program_cost_model->Predict(search_task, *pnow, &pop_scores);

    for (size_t i = 0; i < pnow->size(); ++i) {
      const State& state = (*pnow)[i];
      std::string state_str = state.ToStr();

      if (in_heap.count(state_str) == 0) {
        if (static_cast<int>(heap.size()) < out_size) {
          heap.emplace_back((*pnow)[i], pop_scores[i]);
          std::push_heap(heap.begin(), heap.end(), cmp);
          in_heap.insert(state_str);
        } else if (pop_scores[i] > heap.front().second) {
          std::string old_state_str = heap.front().first.ToStr();
          in_heap.erase(old_state_str);
          in_heap.insert(state_str);

          std::pop_heap(heap.begin(), heap.end(), cmp);
          heap.back() = StateHeapItem(state, pop_scores[i]);
          std::push_heap(heap.begin(), heap.end(), cmp);
        }
        if (pop_scores[i] > max_score) {
          max_score = pop_scores[i];
        }
      }
    }

    // Print statistical information
    if (k % 5 == 0 || k == num_iters) {
      StdCout(verbose) << "GA Iter: " << k;
      if (!heap.empty()) {
        StdCout(verbose) << std::fixed << std::setprecision(4) << "\tMax score: " << max_score
                         << std::fixed << std::setprecision(4)
                         << "\tMin score: " << heap.front().second;
      } else {
        StdCout(verbose) << "\tMax score: N/A\tMin score: N/A";
      }
      StdCout(verbose) << "\t#Pop: " << heap.size() << "\t#M+: " << mutation_success_ct / (k + 1)
                       << "\t#M-: " << mutation_fail_ct / (k + 1) << std::endl;
    }
    if (k == num_iters) {
      break;
    }

    // Compute selection probability
    ComputePrefixSumProb(pop_scores, &pop_selection_probs);

    // TODO(merrymercy, comaniac): add crossover.

    // Do mutation
    while (pnext->size() < population) {
      State tmp_s = (*pnow)[RandomChoose(pop_selection_probs, &rand_gen)];

      if (dis(rand_gen) < mutation_prob) {
        const auto& rule = mutation_rules[RandomChoose(rule_selection_probs, &rand_gen)];
        if (rule->Apply(this, &tmp_s, &rand_gen) == PopulationGenerationRule::ResultKind::kValid) {
          pnext->push_back(std::move(tmp_s));
          mutation_success_ct++;
        } else {
          mutation_fail_ct++;
        }
      } else {
        pnext->push_back(std::move(tmp_s));
      }
    }

    std::swap(pnext, pnow);
    pnext->clear();
  }

  // Copy best states in the heap to out_states
  std::sort(heap.begin(), heap.end(), cmp);
  for (auto& item : heap) {
    best_states.push_back(std::move(item.first));
  }

  double duration = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - tic_begin)
                        .count();
  StdCout(verbose) << "EvolutionarySearch\t\t#s: " << best_states.size()
                   << "\tTime elapsed: " << std::fixed << std::setprecision(2) << duration
                   << std::endl;
  return best_states;
}

Array<MeasureInput> SketchPolicyNode::PickStatesWithEpsGreedy(const Array<State>& best_states,
                                                              const Array<State>& random_states,
                                                              int remaining_n_trials) {
  int num_random =
      static_cast<int>(GetDoubleParam(params, SketchParamKey::eps_greedy) * num_measure_per_iter_);
  int num_good = num_measure_per_iter_ - num_random;

  Array<MeasureInput> inputs;
  size_t offset_best = 0, offset_random = 0;

  while (static_cast<int>(inputs.size()) < std::min(num_measure_per_iter_, remaining_n_trials)) {
    State state;

    bool has_best = offset_best < best_states.size();
    bool has_random = offset_random < random_states.size();

    if (static_cast<int>(inputs.size()) < num_good) {
      // prefer best states
      if (has_best) {
        state = best_states[offset_best++];
      } else if (has_random) {
        state = random_states[offset_random++];
      } else {
        break;
      }
    } else {
      // prefer random states
      if (has_random) {
        state = random_states[offset_random++];
      } else if (has_best) {
        state = best_states[offset_best++];
      } else {
        break;
      }
    }

    // Check if it has already been measured
    std::string state_str = state.ToStr();
    if (!measured_states_set_.count(state_str)) {
      measured_states_set_.insert(std::move(state_str));
      measured_states_vector_.push_back(state);
      inputs.push_back(MeasureInput(search_task, state));
    }
  }

  return inputs;
}


/* convert state to string using tiling size and split meta info
*  @param state : state to be converted
*  @return : string
*/
std::string SketchPolicyNode::state_to_string(const State& state, std::vector<splitMeta*> v_splitMeta_info, const SearchTask& task){
  State ret_state;
  StateNode* pstate;

  if (state->stages.empty()) {
    // If the input state is incomplete with empty operation stage
    // create a new state from init_state and update it first
    ret_state = task->compute_dag->init_state;
    pstate = ret_state.CopyOnWrite();
    pstate->transform_steps = state->transform_steps;
    for (const auto& step : pstate->transform_steps) {
      StepApplyToState(step, &ret_state, task->compute_dag);
    }
  } else {
    ret_state = state;
    pstate = ret_state.CopyOnWrite();
  }

  Array<te::Stage> stages;
  StageToAxesMap stage_to_axes;
  te::Schedule sch;
  Array<te::Tensor> tensors;
  // Replay steps to tvm::Schedule
  std::tie(sch, tensors) = task->compute_dag.ApplySteps(pstate->transform_steps, &stages, &stage_to_axes);
  sch = sch.normalize_for_feature_extraction();
  // Get bound information from TVM schedule
  Map<IterVar, Range> bounds = te::InferBound(sch);

  // Update the state bound information
  for (size_t i = 0; i < pstate->stages.size(); ++i) {
    const Stage& stage = pstate->stages[i];

    if (stage->compute_at == ComputeAtKind::kInlined) {
      continue;
    }

    Array<Iterator> new_iters;
    new_iters.reserve(stage->iters.size());
    // Get bound information from schedule
    // the StageToAxesMap is used to find the corresponding IterVar in TVM schedule result
    for (size_t j = 0; j < stage->iters.size(); ++j) {
      const Iterator& iter = stage->iters[j];
      const IterVar& axis = stage_to_axes.at(stages[i])[j];

      auto find_res = bounds.find(axis);
      if (find_res != bounds.end()) {
        new_iters.push_back(Iterator(iter->name, (*find_res).second, iter->iter_kind,
                                     iter->annotation, &iter->orig_iters));
      } else {
        LOG(FATAL) << "Infer bound fails";
      }
    }

    pstate->stages.Set(
        i, Stage(stage->op, stage->op_type, new_iters, stage->compute_at, stage->attrs));
  }
  const State& init_state = task->compute_dag->init_state;
  std::map<int, int> stage_itr_offset;

  for (auto spm : v_splitMeta_info) {
    int step_id = spm->step_id;
    // // std::cout << "v_splitMeta_info step_id " << step_id  << std::endl;
    auto ps = (state)->transform_steps[step_id].as<SplitStepNode>();
    int orgin_stage_id = ps->stage_id;
    auto ori_iters = (init_state)->stages[orgin_stage_id]->iters;
    // restore iterator id according to transform steps
    if (stage_itr_offset.find(orgin_stage_id) != stage_itr_offset.end()) {
      // accumulate the previous split
      int offset = stage_itr_offset[orgin_stage_id];
      spm->origin_itr = ori_iters[ps->iter_id - offset];

      // Fetch the current tile sizes.
      std::vector<int> lengths(ps->lengths.size() + 1, 1);
      for (int i = 0; i < static_cast<int>(ps->lengths.size()); ++i) {
        lengths[i + 1] = GetIntImm(ps->lengths[i].value());
      }
      lengths[0] = spm->problem_size / ElementProduct(lengths);
      for (auto elm : lengths) {
        spm->add_tilesize(elm);
      }
      stage_itr_offset[orgin_stage_id] += ps->lengths.size();
      if (lengths.size() == 5) {
        spm->parallel = true;
      } else if (lengths.size() == 3) {
        spm->parallel = false;
      } else {
        assert(false && "unknown itr type");
      }
    } else {
      // first one in the stage
      spm->origin_itr = ori_iters[ps->iter_id];
      // Fetch the current tile sizes.
      std::vector<int> lengths(ps->lengths.size() + 1, 1);
      for (int i = 0; i < static_cast<int>(ps->lengths.size()); ++i) {
        lengths[i + 1] = GetIntImm(ps->lengths[i].value());
      }
      lengths[0] = spm->problem_size / ElementProduct(lengths);
      for (auto elm : lengths) {
        spm->add_tilesize(elm);
      }
      stage_itr_offset[orgin_stage_id] = ps->lengths.size();
      if (lengths.size() == 5) {
        spm->parallel = true;
      } else if (lengths.size() == 3) {
        spm->parallel = false;
      } else {
        assert(false && "unknown itr type");
      }
    }
  }
  // std::cout << "------ Enter  state_to_string ----- " << std::endl;
  // if (v_splitMeta_info.empty()) {
  //   std::cout << "v_splitMeta_info is empty" << std::endl;
  // }
  // else{
  //   std::cout << "v_splitMeta_info is not empty" << std::endl;
  // }
  // for (auto spm : v_splitMeta_info) {
  //   std::cout << *spm << std::endl;
  // }
  std::string res = "";
  for (auto spm : v_splitMeta_info) {
    if (spm->parallel == 1) {

      int reg = spm->tile_sizes[1] * spm->tile_sizes[3] * spm->tile_sizes[4];
      int tb = spm->tile_sizes[2];
      int grid = spm->tile_sizes[0];
      // std::cout << "reg = " << reg << " tb = " << tb << " grid = " << grid << std::endl;
      
      // append name+Grid+TB+reg
      res += spm->origin_itr->name + "_Grid" + std::to_string(grid) + "_TB" + std::to_string(tb) + "_reg" + std::to_string(reg);

    } else {
      int outer_SM = spm->tile_sizes[0];
      int inner_outer = spm->tile_sizes[1];
      int inner_inner = spm->tile_sizes[2];
      // std::cout << "outer_SM = " << outer_SM << " inner_outer = " << inner_outer << " inner_inner = " << inner_inner << std::endl;
      
      res += spm->origin_itr->name + "_outer_SM" + std::to_string(outer_SM) + "_inner_outer" + std::to_string(inner_outer) + "_inner_inner" + std::to_string(inner_inner);
    }
    // if (spm == v_splitMeta_info.end()[-1]) {
    //   continue;
    // }
  }
  // std::cout << "res = " << res << std::endl;
  // std::cout << "its state\n" << state << std::endl;
  return res;
}

/* Pack state into MeasureInput and clear measured_states_set_
*  @param best_states: states waiting to be measured
*  @return: MeasureInput
*/
Array<MeasureInput> SketchPolicyNode::PackStateForModel(const Array<State>& best_states, int remaining_n_trials) {
  Array<MeasureInput> inputs;
  size_t offset_best_upperbound = 0;
  size_t offset_best = 0;
  //constrcut all state until no more than remaining_n_trials
  if (best_states.size() > remaining_n_trials){
    offset_best_upperbound = remaining_n_trials;
  }
  else{
    offset_best_upperbound = best_states.size();
  }

  while (offset_best < offset_best_upperbound ){
    State state;
    state = best_states[offset_best++];
    // Check if it has already been measured

    // std::string state_str = state.ToStr();
    std::vector<splitMeta*> v_splitMeta_info;
    v_splitMeta_info = GenerateSplitMeta(this, state);
    std::string state_str = state_to_string(state, v_splitMeta_info, search_task);
    if (!measured_states_set_.count(state_str)) {
      measured_states_set_.insert(std::move(state_str));
      inputs.push_back(MeasureInput(search_task, state));
    }
  }
  measured_states_set_.clear();
  std::cout << "PackState inputs size: " << inputs.size() << std::endl;
  return inputs;
}

/* Pack state into MeasureInput
*  @param best_states: states waiting to be measured
*  @return: MeasureInput
*/
Array<MeasureInput> SketchPolicyNode::PackState(const Array<State>& best_states, int remaining_n_trials) {
  Array<MeasureInput> inputs;
  size_t offset_best_upperbound = 0;
  size_t offset_best = 0;
  //constrcut all state until no more than remaining_n_trials
  if (best_states.size() > remaining_n_trials){
    offset_best_upperbound = remaining_n_trials;
  }
  else{
    offset_best_upperbound = best_states.size();
  }

  while (offset_best < offset_best_upperbound ){
    State state;
    state = best_states[offset_best++];
    // Check if it has already been measured

    // std::string state_str = state.ToStr();
    std::vector<splitMeta*> v_splitMeta_info;
    v_splitMeta_info = GenerateSplitMeta(this, state);
    std::string state_str = state_to_string(state, v_splitMeta_info, search_task);
    if (!measured_states_set_.count(state_str)) {
      measured_states_set_.insert(std::move(state_str));
      inputs.push_back(MeasureInput(search_task, state));
    }
  }
  std::cout << "PackState inputs size: " << inputs.size() << std::endl;
  return inputs;
}

/********** PreloadCustomSketchRule **********/
TVM_REGISTER_OBJECT_TYPE(PreloadCustomSketchRuleNode);

PreloadCustomSketchRule::PreloadCustomSketchRule(PackedFunc meet_condition_func,
                                                 PackedFunc apply_func, String rule_name) {
  auto node = make_object<PreloadCustomSketchRuleNode>();
  node->meet_condition_func = std::move(meet_condition_func);
  node->apply_func = std::move(apply_func);
  node->rule_name = std::move(rule_name);
  data_ = std::move(node);
}

void PreloadCustomSketchRuleNode::Callback(SearchPolicyNode* policy) {
  CHECK(policy->IsInstance<SketchPolicyNode>());
  auto sketch_policy = dynamic_cast<SketchPolicyNode*>(policy);
  sketch_policy->sketch_rules.push_back(
      new RuleCustomSketch(meet_condition_func, apply_func, rule_name));
  StdCout(policy->verbose) << "Custom sketch rule \"" << rule_name << "\" added." << std::endl;
}

TVM_REGISTER_GLOBAL("auto_scheduler.SketchPolicy")
    .set_body_typed([](SearchTask task, CostModel program_cost_model, Map<String, ObjectRef> params,
                       int seed, int verbose,
                       Optional<Array<SearchCallback>> init_search_callbacks) {
      return SketchPolicy(task, program_cost_model, params, seed, verbose, init_search_callbacks);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SketchPolicyGenerateSketches")
    .set_body_typed([](SketchPolicy policy) { return policy->GenerateSketches(); });

TVM_REGISTER_GLOBAL("auto_scheduler.SketchPolicySampleInitialPopulation")
    .set_body_typed([](SketchPolicy policy) {
      const Array<State>& sketches = policy->GenerateSketches();

      Array<State> init_population = policy->SampleInitPopulation(sketches);
      return init_population;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.SketchPolicyEvolutionarySearch")
    .set_body_typed([](SketchPolicy policy, Array<State> init_population, int out_size) {
      Array<State> states = policy->EvolutionarySearch(init_population, out_size);
      return states;
    });

TVM_REGISTER_GLOBAL("auto_scheduler.PrintTitle").set_body_typed([](std::string title) {
  PrintTitle(title, 1);
});

TVM_REGISTER_GLOBAL("auto_scheduler.PreloadCustomSketchRule")
    .set_body_typed([](PackedFunc meet_condition_func, PackedFunc apply_func, String rule_name) {
      return PreloadCustomSketchRule(meet_condition_func, apply_func, rule_name);
    });

}  // namespace auto_scheduler
}  // namespace tvm
