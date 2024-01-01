
#include "sketch_policy_rules.h"

#include <set>
#include <string>
#include <utility>
#include <vector>
#include "sketch_policy.h"
#include "sketch_analysis.h"
#include <tvm/auto_scheduler/compute_dag.h>
#include <iostream>

#include<tvm/auto_scheduler/feature.h>

namespace tvm {
namespace auto_scheduler {

void DataMovementAnalysis::Apply(SketchPolicyNode* policy, State* state) const {

    std::cout << "------------ DVA pass --------------" << std::endl;
    // int skip_first_n_feature_extraction = 0;

    // std::vector<std::vector<float>> features;
    // int max_n_bufs = 5;
    // Array<State> states;
    // states.push_back(*state);

    //GetPerStoreFeaturesFromOneState(*state, policy->search_task, max_n_bufs);

    std::cout << "---------- DVA pass done ------------" << std::endl;

}



}
}
