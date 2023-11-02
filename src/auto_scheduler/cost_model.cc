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
 * \file auto_scheduler/cost_model.cc
 * \brief Cost models that estimate the performance of programs
 */

#include <tvm/auto_scheduler/cost_model.h>
#include <tvm/auto_scheduler/feature.h>

#include <tvm/arith/analyzer.h>
#include <tvm/auto_scheduler/measure.h>
#include <tvm/auto_scheduler/measure_record.h>
#include <tvm/driver/driver_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/parallel_for.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/stmt.h>
#include "search_policy/utils.h"
#include "assert.h"
#include <cuda_runtime.h>

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_OBJECT_TYPE(CostModelNode);
TVM_REGISTER_OBJECT_TYPE(RandomModelNode);
TVM_REGISTER_OBJECT_TYPE(AnaModelNode);
TVM_REGISTER_OBJECT_TYPE(PythonBasedModelNode);

RandomModel::RandomModel() {
  ObjectPtr<RandomModelNode> node = make_object<RandomModelNode>();
  const auto* f = runtime::Registry::Get("auto_scheduler.cost_model.random_fill_float");
  ICHECK(f != nullptr);
  node->random_number_func = reinterpret_cast<const TypedPackedFunc<void(size_t, void*)>*>(f);
  data_ = std::move(node);
}

void RandomModelNode::Update(const Array<MeasureInput>& inputs,
                             const Array<MeasureResult>& results) {}

void RandomModelNode::Predict(const SearchTask& task, const Array<State>& states,
                              std::vector<float>* scores) {
  scores->resize(states.size());
  (*random_number_func)(states.size(), static_cast<void*>(scores->data()));
}

AnaModel::AnaModel() {
  ObjectPtr<AnaModelNode> node = make_object<AnaModelNode>();
  const auto* f = runtime::Registry::Get("auto_scheduler.cost_model.random_fill_float");
  ICHECK(f != nullptr);
  node->random_number_func = reinterpret_cast<const TypedPackedFunc<void(size_t, void*)>*>(f);
  data_ = std::move(node);
}

int getSM() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "Error getting device count: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return -1;
    }

    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    std::string deviceName = deviceProperties.name;

    if (deviceName.find("RTX 3090") != std::string::npos) {
        return 82;
    } else if (deviceName.find("A6000") != std::string::npos) {
        return 84;
    } else if (deviceName.find("A100") != std::string::npos) {
        return 108;
    } else if (deviceName.find("RTX 2080 Ti") != std::string::npos) {
        return 68;
    } else {
        return 68;
    }
}

struct ParallelDataStruct {
  int reg;
  int pz;
  int tb;
};

struct TrueReductionData {
  int sm;
  int pz;
};

struct StencilReductionData {
  int sm;
  int pz;
};

std::map<std::string, ParallelDataStruct> parallel_data_map;
std::map<std::string, TrueReductionData> true_reduction_data_map;
std::map<std::string, StencilReductionData> stencil_reduction_data_map;



class BuffExtractor : public ExprFunctor<double(const PrimExpr& n)> {
 private:
  std::map<std::string, int> sm_name_val_map;

 public:
  BuffExtractor(std::map<std::string, int> sm_name_val_map) {
    this->sm_name_val_map = sm_name_val_map;
  }

  double Extract(const PrimExpr expr) { return this->VisitExpr(expr); }

  double VisitExpr_(const FloatImmNode* op) final {
    // std::cout << "FloatImmNode node "<< op->value  << std::endl;
    return op->value;
  }
  double VisitExpr_(const IntImmNode* op) final {
    // std::cout << "IntImmNode node "  << op->value << std::endl;
    return op->value;
  }

  double VisitExpr_(const VarNode* op) final {
    // std::cout << "var node " << op->name_hint << " " << sm_name_val_map[op->name_hint] <<
    // std::endl;
    std::string v_name = op->name_hint;
    return sm_name_val_map[op->name_hint];
    // if (v_name.find("rx") != std::string::npos || v_name.find("ry") != std::string::npos) {
    //   return (sm_name_val_map[op->name_hint] == 1 ? sm_name_val_map[op->name_hint]
    //                                               : sm_name_val_map[op->name_hint] - 1);
    // } else {
    //   return sm_name_val_map[op->name_hint];
    // }
  }

  double VisitExpr_(const AddNode* op) final {
    // std::cout << "add node "  << std::endl;
    return VisitExpr(op->a) + VisitExpr(op->b);
  }

  double VisitExpr_(const MulNode* op) final {
    // std::cout << "mul node "  << std::endl;
    return VisitExpr(op->a) * VisitExpr(op->b);
  }
};



class ReuseExtractor : public ExprFunctor<void(const PrimExpr& n)> {
 public:
 std::map<std::string, int> name_val_map;
  ReuseExtractor(std::map<std::string, int> name_val_map) {
    this->name_val_map = name_val_map;
  }

  void Extract(const PrimExpr expr) { this->VisitExpr(expr); }

  void VisitExpr_(const FloatImmNode* op) final {
    // std::cout << "FloatImmNode node "<< op->value  << std::endl;
  }
  void VisitExpr_(const IntImmNode* op) final {
    // std::cout << "IntImmNode node "  << op->value << std::endl;
  }

  void VisitExpr_(const VarNode* op) final {
    // std::cout << "var node " << op->name_hint << " set " << sm_name_val_map[op->name_hint] <<
    // std::endl;
    std::string v_name = op->name_hint;
    this->name_val_map[op->name_hint] = 1;

  }

  void VisitExpr_(const AddNode* op) final {
    // std::cout << "add node "  << std::endl;
    VisitExpr(op->a);
    VisitExpr(op->b);

  }

  void VisitExpr_(const MulNode* op) final {
    // std::cout << "mul node "  << std::endl;
    VisitExpr(op->a);
    VisitExpr(op->b);
  }

};

class IndexExtractor : public ExprFunctor<std::string(const PrimExpr& n)> {
 private:
  std::vector<std::string> touch_index;

 public:
  std::vector<std::string> parallel_index;
  std::vector<std::string> reduction_index;
  std::vector<std::string> stencil_index;
  std::vector<std::string> Extract(const PrimExpr expr) {
    auto res = this->VisitExpr(expr);
    if (res != "") touch_index.push_back(res);
    return touch_index;
  }

  std::string VisitExpr_(const FloatImmNode* op) final { return ""; }
  std::string VisitExpr_(const IntImmNode* op) final { return ""; }

  std::string VisitExpr_(const VarNode* op) final {
    // std::cout << "var node " << op->name_hint << " " << sm_name_val_map[op->name_hint] <<
    // std::endl;
    std::string v_name = op->name_hint;
    if (v_name.find("r") != std::string::npos || v_name.find("k") != std::string::npos) {
      if (std::find(reduction_index.begin(), reduction_index.end(),v_name)==reduction_index.end())
        reduction_index.push_back(v_name);
    }
    else{
      if (std::find(parallel_index.begin(), parallel_index.end(),v_name)==parallel_index.end())
        parallel_index.push_back(v_name);
    }

    return v_name;
  }

  std::string VisitExpr_(const AddNode* op) final {
    // std::cout << "add node "  << std::endl;
    auto left = VisitExpr(op->a);
    auto right = VisitExpr(op->b);
    if (left != "") touch_index.push_back(left);
    if (right != "") touch_index.push_back(right);
    if (right != "" && right.find("r") != std::string::npos)  stencil_index.push_back(right);
    return "";
  }

  std::string VisitExpr_(const MulNode* op) final {
    // std::cout << "mul node "  << std::endl;
    auto left = VisitExpr(op->a);
    auto right = VisitExpr(op->b);
    if (left != "") touch_index.push_back(left);
    if (right != "") touch_index.push_back(right);
    return "";
  }
};


std::string strip_itr_name(std::string v) {
  if (v.find(".") != std::string::npos) {
    v = v.substr(0, v.find("."));
  }
  if (v.find("_") != std::string::npos) {
    v = v.substr(0, v.find("_"));
  }
  return v;
}

template<typename T>
std::vector<T> findDiff(std::vector<T> x, std::vector<T> y) {        // no-ref, no-const
    std::vector<T> diff;
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    std::set_difference(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(diff));
    return diff;
}

// std::tuple<int, int, float, float> SketchPolicyNode::extract_features(SketchPolicyNode* policy, State& state, std::vector<splitMeta*> v_splitMeta_info) {
std::tuple<int, int, float, float> AnaModelNode::extract_features(const SearchTask& task, State& state, std::vector<splitMeta*> v_splitMeta_info, std::vector<float> *features) {

  std::cout << "extract_features  " << std::endl;
  std::cout << task->compute_dag << std::endl;
  IndexExtractor index_extract;
  std::string output_FVI;
  std::vector<std::string> par_order_array;
  for (const auto& op : task->compute_dag->ops) {
     if (auto cop = op.as<te::ComputeOpNode>()) {
      if (cop->name.find("temp") == std::string::npos) { 
        output_FVI = cop->axis[cop->axis.size()-1]->var->name_hint;

        for (int i = cop->axis.size()-1; i > -1; i--){
          par_order_array.push_back(cop->axis[i]->var->name_hint);
          // std::cout << "output index order   " << cop->axis[i]->var->name_hint << std::endl;
        }
        //std::cout << cop->name << "----" <<cop->body << std::endl;
        auto ttt = task->compute_dag->access_analyzer->read_from.at(op);
        for (auto p : ttt) {
          //std::cout << "p buffer " << p.first->name << std::endl;
          int reg_buffer_size = 1;
          for (auto vv : p.second) {  // buffer access
            //std::cout << Array<PrimExpr>(vv) << std::endl;
            for (auto indx : vv) {
              index_extract.Extract(indx);
              //std::cout << "indx : " << indx << std::endl;
            }
          }
        }
      }
    }
  }

  //remove stencil from reduction
  std::vector<std::string> true_reduction_index = findDiff(index_extract.reduction_index, index_extract.stencil_index);

  std::cout << "Output FVI : " << output_FVI << std::endl;
  for (auto indx : index_extract.parallel_index){
    std::cout << "paralle indx : " << indx << std::endl;
  }
  for (auto indx : index_extract.stencil_index){
    std::cout << "stencil indx : " << indx << std::endl;
  }
  for (auto indx : true_reduction_index){
    std::cout << "reduction indx : " << indx << std::endl;
  }

  // exit(-1);

  float totalReuse = 0.0;

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

  // for (auto spm : v_splitMeta_info) {
  //   std::cout << *spm << std::endl;
  // }

  // std::cout << "-----------itr id adjust Done------------------------  " << std::endl;

  std::map<std::string, int> sm_name_val_map;
  std::map<std::string, int> reg_novth_name_val_map;
  std::map<std::string, int> reg_name_val_map;

  std::vector<splitMeta*> reg_split_node;
  std::vector<splitMeta*> sm_split_node;
  int grid_size = 1;
  int thread_block_size = 1;
  int registersUsed_per_thread = 1;
  int num_thread_per_block = 1;
  // // std::cout << "end-----------------------------------" << std::endl;
  // std::cout << "end-----------------------------------" << std::endl;
  // int reg_i, reg_j, sm_i, sm_j;
  int reg_ff, reg_xx, reg_yy;
  int sm_rx, sm_ry, sm_rc;
  int pz_rc, pz_rx, pz_ry, pz_ff, pz_xx, pz_yy;
  int tb_xx, tb_yy, tb_ff;

  for (auto spm : v_splitMeta_info) {
    if (spm->parallel == 1) {
      int reg = spm->tile_sizes[1] * spm->tile_sizes[3] * spm->tile_sizes[4];
      int pz = spm->problem_size;
      int tb = spm->tile_sizes[2];
      parallel_data_map[spm->origin_itr->name] = {reg, pz, tb};

      grid_size *= spm->tile_sizes[0];
      thread_block_size *= spm->tile_sizes[2];


      sm_name_val_map[spm->origin_itr->name] =
          spm->tile_sizes[2] * spm->tile_sizes[1] * spm->tile_sizes[3] * spm->tile_sizes[4];
      reg_name_val_map[spm->origin_itr->name] =
          spm->tile_sizes[1] * spm->tile_sizes[3] * spm->tile_sizes[4];
      reg_novth_name_val_map[spm->origin_itr->name] = spm->tile_sizes[3] * spm->tile_sizes[4];
      registersUsed_per_thread *= reg_name_val_map[spm->origin_itr->name];
      num_thread_per_block *= spm->tile_sizes[2];
      reg_split_node.push_back(spm);
    } else {

      int sm_reduction = spm->tile_sizes[1] * spm->tile_sizes[2];
      int pz = spm->problem_size;

      if (spm->origin_itr->name == true_reduction_index[0]){// rc
        true_reduction_data_map[spm->origin_itr->name] = {sm_reduction, pz};
      }
      else {// rx, ry
        stencil_reduction_data_map[spm->origin_itr->name] = {sm_reduction, pz};
      }


      sm_name_val_map[spm->origin_itr->name] = spm->tile_sizes[1] * spm->tile_sizes[2];
      reg_name_val_map[spm->origin_itr->name] = 1;
      sm_split_node.push_back(spm);
    }
    if (spm == v_splitMeta_info.end()[-1]) {
      continue;
    }
    // std::cout << "end-----------------------------------" << std::endl;
  }

  // std::cout << "heuristic pruning-----------------------------------" << std::endl;


  // FVI: xx
  int FVI_reg;
  int FVI_tb;
  int FVI_pz;

  // second_par_indx: yy
  int second_par_reg;
  int second_par_tb;

  // outer_indx: rc
  int outer_reg;
  int outer_tb;
  int outer_pz;

  if (! parallel_data_map.size()){
    for( auto each: parallel_data_map){
      std::cout << "parallel_data_map : " << each.first << " " << each.second.reg << " " << each.second.tb << " " << each.second.pz << std::endl;
    }
  }

  if (! true_reduction_data_map.size()){
    for( auto each: true_reduction_data_map){
      std::cout << "true_reduction_data_map : " << each.first << " " << each.second.sm << " " << each.second.pz << std::endl;
    }
  }

  if (! stencil_reduction_data_map.size()){
    for( auto each: stencil_reduction_data_map){
      std::cout << "stencil_reduction_data_map : " << each.first << " " << each.second.sm << " " << each.second.pz << std::endl;
    }
  }


  // int FVI_flag = 0;
  // // iterate par_order_array to get the tile sizes
  // // par_order_array : [FVI, second_par_indx, outer_indx, nn]
  // for (auto idx_name : par_order_array){
  //   if (idx_name == output_FVI){ // FVI indx
  //     // std::cout << "FVI indx : " << idx_name << std::endl;
  //     //get its reg, tb size and problem size
  //     FVI_reg = parallel_data_map[idx_name].reg;
  //     FVI_tb = parallel_data_map[idx_name].tb;
  //     FVI_pz = parallel_data_map[idx_name].pz;
  //     FVI_flag = 1;
  //   }
  //   else if (FVI_flag == 1){ // second par_index
  //     // std::cout << "second par_index : " << idx_name << std::endl;
  //     //get its reg and tb size
  //     second_par_reg = parallel_data_map[idx_name].reg;
  //     second_par_tb = parallel_data_map[idx_name].tb;
  //     FVI_flag = 0;
  //   }
  //   else{//outer par indx
  //     // std::cout << "outer par indx : " << idx_name << std::endl;
  //     //get its reg, tb size and problem size
  //     int outer_reg = parallel_data_map[idx_name].reg;
  //     int outer_tb = parallel_data_map[idx_name].tb;
  //     int outer_pz = parallel_data_map[idx_name].pz;
  //     break; // no need to iterate outermost index [nn]
  //   }
  // }

  // // avoid high stride write out 
  // // using output index order 
  // // FVI [x], second par [yy], outer [ff]
  // if (FVI_tb * FVI_reg != FVI_pz){
  //   if (second_par_tb/FVI_tb >= 4){
  //     return std::make_tuple(-1, -1, -1, -1);
  //   }
  //   else if (second_par_tb == 1 && outer_tb/FVI_tb >= 4){
  //     return std::make_tuple(-1, -1, -1, -1);
  //   }
  // }

  // //sm production for reduction index
  // float r_sm_production = 1.0;
  // for (auto idx_name : true_reduction_index){
  //   r_sm_production *= true_reduction_data_map[idx_name].sm;
  // }
  // for (auto idx_name : index_extract.stencil_index){
  //   r_sm_production *= stencil_reduction_data_map[idx_name].sm;
  // }

  // if (true_reduction_data_map[true_reduction_index[0]].pz == 3){
  //   // sm filter for stencil index
  //   for (auto idx_name : index_extract.stencil_index){
  //     if (sm_name_val_map[idx_name] != 3){
  //       return std::make_tuple(-1, -1, -1, -1);
  //     }
  //   }
  // }
  // else if (stencil_reduction_data_map[true_reduction_index[0]].pz == 1){// KW/KH, one of the stencil dim problem size == 1
  //   //sm filter for all reduction index
  //   if (r_sm_production < 16 || r_sm_production > 128){
  //     return std::make_tuple(-1, -1, -1, -1);
  //   }
  // }
  // else{
  //   //sm filter for stencil index
  //   for (auto idx_name : index_extract.stencil_index){
  //     if (sm_name_val_map[idx_name] != 3){
  //       return std::make_tuple(-1, -1, -1, -1);
  //     }
  //   }

  //   //sm filter for all reduction index
  //   if (r_sm_production < 16){
  //     return std::make_tuple(-1, -1, -1, -1);
  //   }

  //   //sm filter for true reduction index
  //   if (true_reduction_data_map[true_reduction_index[0]].sm > 64){
  //     return std::make_tuple(-1, -1, -1, -1);
  //   }
  // }

  // //reg filter for FVI and outer index
  // for (auto idx_name : par_order_array){
  //   if (idx_name == output_FVI){// FVI [xx]
  //     if (reg_name_val_map[idx_name] > 30){
  //       return std::make_tuple(-1, -1, -1, -1);
  //     }
  //     FVI_flag = 1;
  //   }
  //   else if (FVI_flag == 1){// second par_index [yy]
  //     FVI_flag = 0;
  //   }
  //   else{// outer par indx [ff]
  //     if (reg_name_val_map[idx_name] > 30){
  //       return std::make_tuple(-1, -1, -1, -1);
  //     }
  //     break;// break after outer index, no need to iterate outermost index [nn]
  //   }
  // }



  // TuningInfo tuning_info;

  // for (auto spm : v_splitMeta_info) {
  //   TuningInfo::DimensionInfo dim_info;

  //   if (spm->parallel == 1) {
  //     dim_info.Gridtile = spm->tile_sizes[0];
  //     dim_info.TBtile = spm->tile_sizes[2];
  //     dim_info.threadcoarse = spm->tile_sizes[1];
  //     dim_info.Reg = spm->tile_sizes[3] * spm->tile_sizes[4];
  //     dim_info.SMneed = spm->tile_sizes[2] * spm->tile_sizes[1] * spm->tile_sizes[3] * spm->tile_sizes[4];
  //   } else {
  //     dim_info.Gridtile = spm->tile_sizes[0];
  //     dim_info.TBtile = spm->tile_sizes[2];
  //     dim_info.threadcoarse = spm->tile_sizes[1];
  //     dim_info.Reg = 1;
  //     dim_info.SMneed = spm->tile_sizes[1] * spm->tile_sizes[2];
  //   }

  //   // Store dimension info in the corresponding dimension in TuningInfo
  //   std::string dim_name = spm->origin_itr->name;
  //   tuning_info.dimension_info[dim_name].push_back(dim_info);

  //   if (spm == v_splitMeta_info.end()[-1]) {
  //     continue;
  //   }
  // }


  // Get CHR and following CA node
  // target_iter_id from CA
  int chr_offest = 0;
  int auto_unroll_max_step = 0;
  std::vector<std::vector<int>> cache_stage_adj_id;
  for (size_t i = (state)->transform_steps.size() - 1; i > 0; i--) {
    if ((state)->transform_steps[i].as<CacheReadStepNode>() != nullptr) {
      auto crs = (state)->transform_steps[i].as<CacheReadStepNode>();
      int target_stage_id = crs->stage_id + 1 + chr_offest;

      // getting "CA" --->> ad hoc
      auto ca = (state)->transform_steps[i + 1].as<ComputeAtStepNode>();
      int append_itr_position = ca->target_iter_id;
      int ca_target_stage_id = ca->target_stage_id + chr_offest;
      // // std::cout << target_stage_id << "--" << ca_target_stage_id << " " << append_itr_position
      //           << std::endl;
      cache_stage_adj_id.push_back({target_stage_id, ca_target_stage_id, append_itr_position});
      chr_offest++;
    }

    if ((state)->transform_steps[i].as<PragmaStepNode>() != nullptr) {
      auto psn = (state)->transform_steps[i].as<PragmaStepNode>();
      size_t pos = 0;
      for (; pos < psn->pragma_type.size(); ++pos) {
        if ((*( psn->pragma_type.c_str() + pos)) == '$') {
          break;
        }
      }
      auto_unroll_max_step = atoi(psn->pragma_type.c_str() + pos + 1);
      // std::cout << "unrll factor " << auto_unroll_max_step  << std::endl;

    }


  }
  // std::cout << "-------------CHR CA done----------------------" << std::endl;

  std::unordered_set<std::string> index_set;
  // TODO: what if we have more than one compute  ??
  std::map<std::string, std::unordered_set<std::string>> buf_index_map;

  for (const auto& op : task->compute_dag->ops) {
    if (auto cop = op.as<te::ComputeOpNode>()) {
      // // std::cout << "------SM alloca ----------" << cop  << std::endl;
      if (cop->name.find("temp") == std::string::npos) {
        // output
        Array<PrimExpr> output_index;
        for (auto i = 0; i < cop->axis.size(); i++) {
          output_index.push_back(PrimExpr(cop->axis[i].get()->var));
        }
        for (auto indx : output_index) {
          IndexExtractor extractor_indx;
          std::vector<std::string> touch_index = extractor_indx.Extract(indx);
          for (auto ii : touch_index) {
            index_set.insert(ii);
          }
        }
        buf_index_map[cop->name] = index_set;

        // inputs
        auto ttt = task->compute_dag->access_analyzer->read_from.at(op);
        for (auto p : ttt) {
          for (auto vv : p.second) {  // buffer access
            index_set.clear();
            for (auto indx : vv) {
              IndexExtractor extractor_indx;
              std::vector<std::string> touch_index = extractor_indx.Extract(indx);
              for (auto ii : touch_index) {
                index_set.insert(ii);
              }
            }
            buf_index_map[p.first->name] = index_set;
          }
        }
      }
    }
  }

  //// check buffer index information
   for (auto itr : buf_index_map){
     // std::cout << itr.first << std::endl;
     for (auto el : itr.second){
        // std::cout << el << " ";
     }
     // std::cout << std::endl;
   }

   //(Approximation) op access index info from base computation
  int totalShared = 0;
  BuffExtractor extractor(sm_name_val_map);

  for (const auto& op : task->compute_dag->ops) {
    if (auto cop = op.as<te::ComputeOpNode>()) {
      // // std::cout << "------SM alloca ----------" << cop  << std::endl;
      if (cop->name.find("temp") == std::string::npos) {
        auto ttt = task->compute_dag->access_analyzer->read_from.at(op);
        for (auto p : ttt) {
          // std::cout << "p buffer " << p.first->name << std::endl;
          int sm_buffer_size = 1;
          // int reuse_factor = 1;
          for (auto vv : p.second) {  // buffer access
            // std::cout << Array<PrimExpr>(vv) << std::endl;
            for (auto indx : vv) {
              auto index_extent = extractor.Extract(indx);
              // std::cout << "indx : " << indx << " " << index_extent << std::endl;
              sm_buffer_size *= index_extent;
            }
          }
          // std::cout << "buffer size " << sm_buffer_size << std::endl << std::endl;
          totalShared += sm_buffer_size;
        }
      }
    }
  }
  // std::cout << 
  // std::cout << "---------------SM alloction done--------------------" << " total shared: " << totalShared << std::endl;

  int totalReg = 1;
  BuffExtractor extractor_reg(reg_name_val_map);

  // ad hoc for output
  for (auto itr : reg_name_val_map){
    //std::cout << "R  " <<  itr.first << "----" << itr.second<< std::endl;
    totalReg *= itr.second;
  }

  for (const auto& op : task->compute_dag->ops) {
    if (auto cop = op.as<te::ComputeOpNode>()) {
      if (cop->name.find("temp") != std::string::npos) continue;
      if (cop->name.find("local") == std::string::npos) { // Yufan:: why to use "local" as keyword??
        //std::cout << cop->name << "----" <<cop->body << std::endl;

        auto ttt = task->compute_dag->access_analyzer->read_from.at(op);

        for (auto p : ttt) {
          ///std::cout << "p buffer " << p.first->name << std::endl;
          int reg_buffer_size = 1;
          for (auto vv : p.second) {  // buffer access
            // std::cout << Array<PrimExpr>(vv) << std::endl;
            for (auto indx : vv) {
              auto index_extent = extractor_reg.Extract(indx);
              // std::cout << "indx : " << indx << " " << index_extent << std::endl;
              reg_buffer_size *= index_extent;
            }
          }
          //std::cout << "reg buffer size "  << reg_buffer_size << std::endl << std::endl;
          totalReg += reg_buffer_size;
        }
      }
    }
  }

  // std::cout << "------reg alloca DONE ---------- " << " total reg: " << totalReg << std::endl;

  std::vector<std::pair<std::string, int>> loops_inandout;
  auto temp_map = reg_novth_name_val_map;
  // Registerlevel reuse --> get outer loop structure
  for (int i = 0; i < (ret_state)->stages.size(); i++) {
    auto stg = (ret_state)->stages[i];

    if (stg.get()->op->name.find("local") != std::string::npos) {
      // extract all loops structure
      for (size_t i = stg->iters.size()-1; i > 0; i--) {
        const Iterator& iter = stg->iters[i];
        if (static_cast<int>(iter->annotation) == 0) {
          // std::cout << iter->name << " (" << iter->range->min << "," << iter->range->extent << ")"
          //           << std::endl;
          auto itvn = strip_itr_name(iter->name);
          int extent = GetIntImm(iter->range->extent);
          //kncoking out reg loop

          //std::cout << itvn << "  "<<temp_map[itvn] << "?" << extent<< std::endl;


          if (temp_map[itvn] / extent >= 1){
              //std::cout << "knocking" << std::endl;
              temp_map[itvn] = temp_map[itvn]/ extent;
          }
          else{
            loops_inandout.push_back(std::make_pair(itvn, extent));
          }
        }
      }
    }
  }


  // std::cout << "\n--------------compute reg buffer reuse ---------------------" << std::endl;
  int reg_alloc = 1;
  for (auto itr : reg_name_val_map){
    reg_alloc *= itr.second;
  }

  // if (reg_alloc > 256){
  //   std::cout << "WARNING! not pure REG " << reg_alloc << std::endl;
  // }

  std::map<std::string, int> buf_reuse_map;

  for (const auto& op : task->compute_dag->ops) {
    if (auto cop = op.as<te::ComputeOpNode>()) {
      if (cop->name.find("temp") == std::string::npos) {
        // output
        Array<PrimExpr> output_index;
        ReuseExtractor reuse_extractor(reg_name_val_map);
        for (auto i = 0; i < cop->axis.size(); i++) {
          output_index.push_back(PrimExpr(cop->axis[i].get()->var));
        }
        //std::cout << output_index << std::endl;

        for (auto indx : output_index) {
          reuse_extractor.Extract(indx);
        }
        int reuse_factor = 1;
        for (auto itr : reuse_extractor.name_val_map){
            reuse_factor *= itr.second;
        }
        //std::cout << "reg " << cop->name << ", reuse_factor, " << reuse_factor << std::endl;
        buf_reuse_map[cop->name] = reuse_factor;



        auto ttt = task->compute_dag.operator->()->access_analyzer->read_from.at(op);
        for (auto p : ttt) {
          ReuseExtractor reuse_extractor(reg_name_val_map);
          int reuse_factor = 1;
          for (auto vv : p.second) {  // buffer access
            //std::cout << Array<PrimExpr>(vv) << std::endl;
            for (auto indx : vv) {
              reuse_extractor.Extract(indx);
            }
          }
          for (auto itr : reuse_extractor.name_val_map){
            reuse_factor *= itr.second;
          }
          //std::cout << "reg " << p.first->name << ", reuse_factor, " << reuse_factor << std::endl;
          buf_reuse_map[p.first->name] = reuse_factor;
        }
      }
    }
  }


  for (auto itr : buf_index_map) {
    int reuse_factor = buf_reuse_map[itr.first];
    for (auto loop = loops_inandout.begin(); loop != loops_inandout.end(); loop++){
      std::string loop_itr_name = std::get<0>(*loop);
      int loop_extent = std::get<1>(*loop);
      std::unordered_set<std::string> index_set = itr.second;
      if (index_set.find(loop_itr_name) == index_set.end()){
        //std::cout << "loop itr " << loop_itr_name << std::endl;
        reuse_factor *= loop_extent;
        //std::cout << "reuse " << std::endl;
      }
      else if (loop_extent == 1){
        continue;
      }
      else{
        break;
      }
    }
    // std::cout << itr.first << ", reuse_factor, " << reuse_factor << std::endl;
    totalReuse += 1.0/float(reuse_factor);
  }


  // std::cout << "\n--------------compute sm buffer reuse ---------------------" << std::endl;
  for (const auto& op : task->compute_dag->ops) {
    if (auto cop = op.as<te::ComputeOpNode>()) {
      // std::cout << "------SM alloca ----------" << cop  << std::endl;
      if (cop->name.find("temp") == std::string::npos) {
        auto ttt = task->compute_dag.operator->()->access_analyzer->read_from.at(op);
        for (auto p : ttt) {
          ReuseExtractor reuse_extractor(sm_name_val_map);
          int reuse_factor = 1;
          for (auto vv : p.second) {  // buffer access
            //std::cout << Array<PrimExpr>(vv) << std::endl;
            for (auto indx : vv) {
              reuse_extractor.Extract(indx);
            }
          }
          for (auto itr : reuse_extractor.name_val_map){
            reuse_factor *= itr.second;
          }
          // std::cout << "SM" << p.first->name << ", reuse_factor, " << reuse_factor << std::endl;
        }
      }
    }
  }

  totalReuse = 1.0/totalReuse;
  int num_sm = getSM();
  // std::cout << "\nThe GPU's SM number is " << num_sm << std::endl;

  float wave_fac = grid_size*1.0/num_sm;
  float wave = std::ceil(wave_fac);
  float wave_efficiency = wave_fac/wave;

  // TODO: change to 0.67
  // if (thread_block_size < 32 || thread_block_size > 1024 || wave_efficiency < 0.67){
  //   return std::make_tuple(-1, -1, -1, -1);
  // }
  if (thread_block_size < 32 || thread_block_size > 1024){
    return std::make_tuple(-1, -1, -1, -1);
  }
  // if (thread_block_size > 1024){
  //   return std::make_tuple(-1, -1, -1, -1);
  // }

  return std::make_tuple(totalShared, totalReg, totalReuse, wave_efficiency);
}

void AnaModelNode::Update(const Array<MeasureInput>& inputs,
                             const Array<MeasureResult>& results) {}

void AnaModelNode::Predict(const SearchTask& task, const Array<State>& states,
                              std::vector<float>* scores) {
  std::cout << "AnaModelNode::Predict" << std::endl;
  std::cout << "size of states " << states.size() << std::endl;
  std::cout << "size of scores " << scores->size() << std::endl;

  scores->resize(states.size());
  std::cout << "size of states " << states.size() << std::endl;
  std::cout << "size of scores " << scores->size() << std::endl;
  int max_n_bufs = 5;
  std::vector<std::vector<float>> features;
  auto_scheduler::GetPerStoreFeaturesFromStates(states, task, 0, max_n_bufs, &features);


  //features[1] -> global_trans
  //features[2] -> shared_trans
  //features[3] -> est_occupancy

  // par for to get the features from extract_features for each state in states

  // features tuple: ['wave_efficiency', 'est_occupancy', 'ILP', 'WLP, 'Concurrent_estimate', 'totalReuse', 'OI_Global'].
  // support::parallel_for(0, states.size(), [&](int index) {
  //   std::vector<float> features_extracted;
  //   features_extracted.reserve(5);

  //   //call extract_features


  // });

  

  for (auto feature: features){
    std::cout << "feature size " << feature.size() << std::endl;
    // assert( feature.size() != 0 );
    // std::cout << "fea[0]" << feature[0] << std::endl;   
    // std::cout << "fea[1]" << feature[1] << std::endl;   
    // std::cout << "fea[2]" << feature[2] << std::endl;   
    // std::cout << "fea[3]" << feature[3] << std::endl;
    // std::cout << "fea[4]" << feature[4] << std::endl;    
  }

  for (auto i = 0; i < states.size(); i++){
    std::cout << "GetPerStoreFeaturesFromStates feate "<< i << " - " <<features[i].size() << std::endl;
    if (features[i].size() == 0){
      //Failure case, gives a very small score
      (*scores)[i] = -1e-10f;
    }else{
      (*scores)[i] = features.at(i).at(2);
    }

    for (auto ii : features[i]){
      std::cout << "elel " << ii << std::endl;
    }
  } 

  for (auto i = 0; i < states.size(); i++){
    std::cout << "scores "<< scores->at(i) << std::endl;
  }
  /***
   * TODO: normalized  inverse ??
   * */
}

PythonBasedModel::PythonBasedModel(PackedFunc update_func, PackedFunc predict_func,
                                   PackedFunc predict_stage_func) {
  auto node = make_object<PythonBasedModelNode>();
  node->update_func = std::move(update_func);
  node->predict_func = std::move(predict_func);
  node->predict_stage_func = std::move(predict_stage_func);
  data_ = std::move(node);
}

void PythonBasedModelNode::Update(const Array<MeasureInput>& inputs,
                                  const Array<MeasureResult>& results) {
  update_func(inputs, results);
}

void PythonBasedModelNode::Predict(const SearchTask& task, const Array<State>& states,
                                   std::vector<float>* scores) {
  scores->resize(states.size());
  predict_func(task, states, static_cast<void*>(scores->data()));
}

void PythonBasedModelNode::PredictStages(const SearchTask& task, const Array<State>& states,
                                         std::vector<float>* state_scores,
                                         std::vector<std::vector<float>>* stage_scores) {
  size_t n_states = states.size();
  size_t n_stages = task->compute_dag->init_state->stages.size();
  std::vector<float> flatten_scores;
  // Allocate sufficient spaces.
  flatten_scores.resize(n_states * n_stages * 2);
  predict_stage_func(task, states, static_cast<void*>(flatten_scores.data()));

  /* For faster data copy between c++ and python, the python part returns scores in a
   * single flatten array using a packed format. The c++ part then unpacks the flatten array.
   *
   * The packed format is:
   * {
   *   float  scores[N];                 // scores[i] is the score for states[i].
   *   int    n_stage_0;                 // the number of stages in states[0]
   *   float  stage_scores_0[[n_stage_0] // the scores for all stages in states[0]
   *   int    n_stage_1;                 // the number of stages in states[1]
   *   float  stage_scores_1[n_stage_1]; // the scores for all stages in states[1]
   *   ...
   *   int    n_stage_i;                 // the number of stages in states[i]
   *   float  stage_scores_1[n_stage_i]; // the scores for all stages in states[i]
   *   ...  // until i == N - 1
   * }
   * To implement this format, we also store int as float, so we can store all numbers
   * into a single float array.
   */

  // Unpack flatten scores.
  state_scores->clear();
  stage_scores->clear();

  // Score of each states.
  for (size_t i = 0; i < n_states; ++i) {
    state_scores->push_back(flatten_scores[i]);
  }

  // Score of each stage in each states.
  size_t idx = n_states;
  for (size_t i = 0; i < n_states; ++i) {
    ICHECK_LE(idx, flatten_scores.size());

    // Number of scored stages of this state.
    int s_length = static_cast<int>(flatten_scores[idx++]);

    if (s_length > 0) {
      std::vector<float> scores;
      int offset = 0;

      if ((*state_scores)[i] > -INFINITY) {
        // If the score is valid. Copy scored stages and assign 0 to placeholder
        // and inlined stages. If the score is 0, meaning this state failed to
        // be lowered. Just bypass to update offset.
        for (const Stage& stage : states[i]->stages) {
          if (stage->op_type == StageKind::kPlaceholder) {
            scores.push_back(0);
            continue;
          }
          if (stage->compute_at == ComputeAtKind::kInlined) {
            scores.push_back(0);
            continue;
          }
          scores.push_back(flatten_scores[idx + offset]);
          offset++;
        }
        ICHECK_EQ(offset, s_length);
        stage_scores->push_back(std::move(scores));
      }
      idx += s_length;
    } else {
      // Cost model does not provide any stage score details.
      stage_scores->push_back({});
    }
  }
}

TVM_REGISTER_GLOBAL("auto_scheduler.RandomModel").set_body_typed([]() { return RandomModel(); });

TVM_REGISTER_GLOBAL("auto_scheduler.AnaModel").set_body_typed([]() { return AnaModel(); });

TVM_REGISTER_GLOBAL("auto_scheduler.PythonBasedModel")
    .set_body_typed([](PackedFunc update_func, PackedFunc predict_func,
                       PackedFunc predict_stage_func) {
      return PythonBasedModel(update_func, predict_func, predict_stage_func);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.CostModelUpdate")
    .set_body_typed([](CostModel model, Array<MeasureInput> inputs, Array<MeasureResult> results) {
      model->Update(inputs, results);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.CostModelPredict")
    .set_body_typed([](CostModel model, SearchTask task, Array<State> states) {
      std::vector<float> scores;
      model->Predict(task, states, &scores);
      Array<FloatImm> ret;
      for (auto x : scores) {
        std::cout << "score " << x << std::endl;
        ret.push_back(FloatImm(DataType::Float(32), x));
      }
      return ret;
    });

}  // namespace auto_scheduler
}  // namespace tvm
