#include <map>
#include <numeric>

#include <Eigen/Dense>

#include <bingocpp/agraph/simplification_backend/simplification_backend.h>
#include <bingocpp/agraph/constants.h>
#include <bingocpp/agraph/operator_definitions.h>

namespace bingo {
namespace simplification_backend {

std::vector<bool> GetUtilizedCommands(const Eigen::ArrayX3i &stack) {
  std::vector<bool> used_commands(stack.rows());
  used_commands.back() = true;
  int stack_size = stack.rows();
  for (int i = 1; i < stack_size; i++) {
    int row = stack_size - i;
    int node = stack(row, kOpIdx);
    int param1 = stack(row, kParam1Idx);
    int param2 = stack(row, kParam2Idx);
    if (used_commands[row] && node > Op::kConstant) {
      used_commands[param1] = true;
      if (AGraph::HasArityTwo(node)) {
        used_commands[param2] = true;
      }
    }
  }
  return used_commands;
}

Eigen::ArrayX3i SimplifyStack(const Eigen::ArrayX3i &stack) {
  std::vector<bool> used_command = GetUtilizedCommands(stack);
  std::map<int, int> reduced_param_map;
  int num_commands = 0;
  num_commands = std::accumulate(used_command.begin(), used_command.end(), 0);
  Eigen::ArrayX3i new_stack(num_commands, 3);

  for (int i = 0, j = 0; i < stack.rows(); ++i) {
    if (used_command[i]) {
      new_stack(j, kOpIdx) = stack(i, kOpIdx);
      if (AGraph::IsTerminal(new_stack(j, kOpIdx))) {
        new_stack(j, kParam1Idx) = stack(i, kParam1Idx);
        new_stack(j, kParam2Idx) = stack(i, kParam2Idx);
      } else {
        new_stack(j, kParam1Idx) = reduced_param_map[stack(i, kParam1Idx)];
        if (AGraph::HasArityTwo(new_stack(j, kOpIdx))) {
          new_stack(j, kParam2Idx) = reduced_param_map[stack(i, kParam2Idx)];
        } else {
          new_stack(j, kParam2Idx) = new_stack(j, kParam1Idx);
        }
      }
      reduced_param_map[i] = j;
      ++j;
    }
  }
  return new_stack;
}

} // namespace simplification_backend
} // namespace bingo 
