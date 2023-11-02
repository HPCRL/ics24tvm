#ifndef TVM_AUTO_SCHEDULER_SEARCH_POLICY_SKETCH_ANALYSIS_H_
#define TVM_AUTO_SCHEDULER_SEARCH_POLICY_SKETCH_ANALYSIS_H_

#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/search_task.h>

#include <string>
#include <utility>
#include <vector>

#include "utils.h"

namespace tvm {
namespace auto_scheduler {
class SketchPolicyNode;

class SketchAnalysisRule {
 public:
  virtual void Apply(SketchPolicyNode* policy, State* state) const = 0;
  virtual ~SketchAnalysisRule() = default;
};

#define DEFINE_SK_RULE(rule_name)                                                    \
  class rule_name : public SketchAnalysisRule {                                             \
   public:                                                                                        \
    void Apply(SketchPolicyNode* policy, State* state) const final; \
  };

DEFINE_SK_RULE(DataMovementAnalysis);

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_SEARCH_POLICY_SKETCH_ANALYSIS_H_
