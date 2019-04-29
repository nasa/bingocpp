#include <map>
#include <numeric>

#include <Eigen/Dense>

#include "BingoCpp/acyclic_graph.h"
#include "BingoCpp/backend.h"
#include "BingoCpp/backend_nodes.h"

const int NODE_IDX = 0;
const int OP_1 = 1;
const int OP_2 = 2;

namespace bingo {
namespace backend {
namespace {

Eigen::ArrayXXd reverse_eval(const std::pair<int, int>& deriv_shape,
                             const int deriv_wrt_node,
                             const std::vector<Eigen::ArrayXXd>& forward_eval,
                             const Eigen::ArrayX3i& stack) {
  int num_samples = deriv_shape.first;
  int num_features = deriv_shape.second;
  int stack_depth = stack.rows();

  Eigen::ArrayXXd derivative = Eigen::ArrayXXd::Zero(num_samples, num_features);
  std::vector<Eigen::ArrayXXd> reverse_eval(stack_depth); 
  for (int row = 0; row < stack_depth; row++) {
      reverse_eval[row] = Eigen::ArrayXd::Zero(num_samples);
  }

  reverse_eval[stack_depth-1] = Eigen::ArrayXd::Ones(num_samples);
  for (int i = stack_depth - 1; i >= 0; i--) {
    int node = stack(i, NODE_IDX);
    int param1 = stack(i, OP_1);
    int param2 = stack(i, OP_2);
    if (node == deriv_wrt_node) {
      derivative.col(param1) += reverse_eval[i];
    } else {
      reverse_eval_function(node, i, param1, param2, forward_eval, reverse_eval);
    }
  }
  return derivative;
}

Eigen::ArrayXXd reverse_eval_with_mask(const std::pair<int, int>& deriv_shape,
                                       const int deriv_wrt_node,
                                       const std::vector<Eigen::ArrayXXd>& forward_eval,
                                       const Eigen::ArrayX3i& stack,
                                       const std::vector<bool>& mask) {
  int num_samples = deriv_shape.first;
  int num_features = deriv_shape.second;
  int stack_depth = stack.rows();

  Eigen::ArrayXXd derivative = Eigen::ArrayXXd::Zero(num_samples, num_features);
  std::vector<Eigen::ArrayXXd> reverse_eval(stack_depth); 
  for (int row = 0; row < stack_depth; row++) {
    if (mask[row]) {
      reverse_eval[row] = Eigen::ArrayXd::Zero(num_samples);
    }
  }

  reverse_eval[stack_depth-1] = Eigen::ArrayXd::Ones(num_samples);
  for (int i = stack_depth - 1; i >= 0; i--) {
    if (mask[i]) {
      int node = stack(i, NODE_IDX);
      int param1 = stack(i, OP_1);
      int param2 = stack(i, OP_2);
      if (node == deriv_wrt_node) {
        derivative.col(param1) += reverse_eval[i];
      } else {
        reverse_eval_function(node, i, param1, param2, forward_eval, reverse_eval);
      }
    }
  }
  return derivative;
}

std::vector<Eigen::ArrayXXd> forward_eval(
    const Eigen::ArrayX3i& stack,
    const Eigen::ArrayXXd& x,
    const Eigen::VectorXd& constants) {
  std::vector<Eigen::ArrayXXd> _forward_eval(stack.rows());

  for (int i = 0; i < stack.rows(); ++i) {
    int node = stack(i, NODE_IDX);
    int op1 = stack(i, OP_1);
    int op2 = stack(i, OP_2);
    _forward_eval[i] = forward_eval_function(
      node, op1, op2, x, constants, _forward_eval);
  }
  return _forward_eval;
}

std::vector<Eigen::ArrayXXd> forward_eval_with_mask(
    const Eigen::ArrayX3i& stack,
    const Eigen::ArrayXXd& x,
    const Eigen::VectorXd& constants,
    const std::vector<bool>& mask) {
  std::vector<Eigen::ArrayXXd> _forward_eval(stack.rows());

  for (int i = 0; i < stack.rows(); ++i) {
    if (mask[i]) {
      int node = stack(i, NODE_IDX);
      int op1 = stack(i, OP_1);
      int op2 = stack(i, OP_2);
      _forward_eval[i] = forward_eval_function(
        node, op1, op2, x, constants, _forward_eval);
    }
  }
  return _forward_eval;
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> evaluate_with_derivative(
    const Eigen::ArrayX3i& stack,
    const Eigen::ArrayXXd& x,
    const Eigen::VectorXd& constants,
    const bool param_x_or_c) {
  std::vector<Eigen::ArrayXXd> _forward_eval = forward_eval(
      stack, x, constants);

  std::pair<int, int> deriv_shape;
  int deriv_wrt_node;
  if (param_x_or_c) {  // true = x
    deriv_shape = std::make_pair(x.rows(), x.cols());
    deriv_wrt_node = 0;
  } else {  // false = c
    deriv_shape = std::make_pair(x.rows(), constants.size());
    deriv_wrt_node = 1;
  }

  Eigen::ArrayXXd derivative = reverse_eval(
      deriv_shape, deriv_wrt_node, _forward_eval, stack);
  return std::make_pair(_forward_eval.back(), derivative);
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> evaluate_with_derivative_and_mask(
    const Eigen::ArrayX3i& stack,
    const Eigen::ArrayXXd& x,
    const Eigen::VectorXd& constants,
    const std::vector<bool>& mask,
    const bool param_x_or_c) {
  std::vector<Eigen::ArrayXXd> forward_eval = forward_eval_with_mask(
      stack, x, constants, mask);

  std::pair<int, int> deriv_shape;
  int deriv_wrt_node;
  if (param_x_or_c) {  // true = x
    deriv_shape = std::make_pair(x.rows(), x.cols());
    deriv_wrt_node = 0;
  } else {  // false = c
    deriv_shape = std::make_pair(x.rows(), constants.size());
    deriv_wrt_node = 1;
  }

  Eigen::ArrayXXd derivative = reverse_eval_with_mask(
      deriv_shape, deriv_wrt_node, forward_eval, stack, mask);
  return std::make_pair(forward_eval.back(), derivative);
}
} // namespace

bool isCpp() {
    return true;
}

Eigen::ArrayXXd evaluate(const Eigen::ArrayX3i& stack,
                         const Eigen::ArrayXXd& x,
                         const Eigen::VectorXd& constants) {
  std::vector<Eigen::ArrayXXd> _forward_eval = forward_eval(
      stack, x, constants);
  return _forward_eval.back();  
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> evaluateWithDerivative(
    const Eigen::ArrayX3i& stack,
    const Eigen::ArrayXXd& x,
    const Eigen::VectorXd& constants,
    const bool param_x_or_c) {
  return evaluate_with_derivative(
    stack, x, constants, param_x_or_c);
}

Eigen::ArrayXXd simplifyAndEvaluate(const Eigen::ArrayX3i& stack,
                                    const Eigen::ArrayXXd& x,
                                    const Eigen::VectorXd& constants) {
  std::vector<bool> mask = getUtilizedCommands(stack);
  std::vector<Eigen::ArrayXXd> forward_eval = forward_eval_with_mask(
      stack, x, constants, mask);
  return forward_eval.back();
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> simplifyAndEvaluateWithDerivative(
    const Eigen::ArrayX3i& stack,
    const Eigen::ArrayXXd& x,
    const Eigen::VectorXd& constants,
    const bool param_x_or_c) {
  std::vector<bool> mask = getUtilizedCommands(stack);
  return evaluate_with_derivative_and_mask(stack, x, constants, mask, param_x_or_c);
}

std::vector<bool> getUtilizedCommands(const Eigen::ArrayX3i& stack) {
  std::vector<bool> used_commands(stack.rows());
  used_commands.back() = true;
  int stack_size = stack.rows();
  for (int i = 1; i < stack_size; i++) {
    int row = stack_size - i;
    int node = stack(row, NODE_IDX);
    int param1 = stack(row, OP_1);
    int param2 = stack(row, OP_2);
    if (used_commands[row] && node > 1) {
      used_commands[param1] = true;
      if (AcyclicGraph::has_arity_two(node)) {
        used_commands[param2] = true;
      }
    }
  }
  return used_commands;
}

Eigen::ArrayX3i simplifyStack(const Eigen::ArrayX3i& stack) {
  std::vector<bool> used_command = getUtilizedCommands(stack);
  std::map<int, int> reduced_param_map;
  int num_commands = 0;
  num_commands = std::accumulate(used_command.begin(), used_command.end(), 0);
  Eigen::ArrayX3i new_stack(num_commands, 3);

  for (int i = 0, j = 0; i < stack.rows(); ++i) {
    if (used_command[i]) {
      new_stack(j, 0) = stack(i, 0);
      if (AcyclicGraph::is_terminal(new_stack(j, 0))) {
        new_stack(j, 1) = stack(i, 1);
        new_stack(j, 2) = stack(i, 2);
      } else {
        new_stack(j, 1) = reduced_param_map[stack(i, 1)];
        if (AcyclicGraph::has_arity_two(new_stack(j, 0))) {
          new_stack(j, 2) = reduced_param_map[stack(i, 2)];
        } else {
          new_stack(j, 2) = new_stack(j, 1);
        }
      }
      reduced_param_map[i] = j;
      ++j;
    }
  }
  return new_stack;
}

int get_arity(int node) {
  if (AcyclicGraph::is_terminal(node)) return 0;
  return AcyclicGraph::has_arity_two(node) ? 2 : 1;
}
} // namespace backend
} // namespace bingo 
