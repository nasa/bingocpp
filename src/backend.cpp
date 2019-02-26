#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <map>

#include <Eigen/Dense>

#include "BingoCpp/acyclic_graph.h"
#include "BingoCpp/acyclic_graph_nodes.h"
#include "BingoCpp/backend.h"
#include "BingoCpp/backend_nodes.h"



const int NODE_IDX = 0;
const int OP_1 = 1;
const int OP_2 = 2;
const int BACKENDNODES = 0;

bool IsCpp() {
    return true;
}

int get_arity(int node) {
  if (AcyclicGraph::is_terminal(node)) return 0;
  return AcyclicGraph::has_arity_two(node) ? 2 : 1;
}

void ReverseSingleCommand(const Eigen::ArrayX3i &stack,
                          const int command_index,
                          const std::vector<Eigen::ArrayXXd> &forward_buffer,
                          std::vector<Eigen::ArrayXXd> &reverse_buffer,
                          const std::set<int> &dependencies) {
  // Computes reverse autodiff partial of a stack command.
  for (auto const& dependency : dependencies) {
    agraphnodes::derivative_eval_function(stack(dependency, 0), stack, 
                                           command_index, forward_buffer,
                                           reverse_buffer, dependency);
  }
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> agraph_nodes_deriv(
  const Eigen::ArrayX3i &stack,
  const Eigen::ArrayXXd &x,
  const Eigen::VectorXd &constants,
  const std::vector<bool> &mask,
  const bool param_x_or_c) {
  
  // Evaluates a stack and its derivative with the given x and constants.
  std::vector<Eigen::ArrayXXd> forward_eval(stack.rows());
  std::vector<std::set<int>> stack_dependencies(stack.rows(),
                          std::set<int>());
  int deriv_size;
  int deriv_operator_number;

  if (param_x_or_c) {  // true = x
    deriv_size = x.cols();
    deriv_operator_number = 0;

  } else {  // false = c
    deriv_size = constants.size();
    deriv_operator_number = 1;
  }

  std::vector<std::set<int>> param_dependencies(deriv_size, std::set<int>());

  // forward eval with dependencies
  for (std::size_t i = 0; i < stack.rows(); ++i) {
    if (mask[i]) {
      agraphnodes::forward_eval_function(stack(i, 0), stack, 
                                          x, constants, forward_eval, i);
      if (stack(i, 0) == deriv_operator_number) {
        param_dependencies[stack(i, 1)].insert(i);
      }
      for (int j = 0; j < get_arity(stack(i, 0)); ++j) {
        stack_dependencies[stack(i, j + 1)].insert(i);
      }
    }
  }
  // reverse pass through stack
  std::vector<Eigen::ArrayXXd> reverse_eval(stack.rows());
  reverse_eval[stack.rows() - 1] = Eigen::ArrayXXd::Ones(x.rows(), 1);
  for (int i = stack.rows() - 2; i >= 0; --i) {
    if (mask[i]) {
      reverse_eval[i] = Eigen::ArrayXXd::Zero(x.rows(), 1);
      ReverseSingleCommand(stack, i, forward_eval, reverse_eval,
                           stack_dependencies[i]);
    }
  }
  // build derivative array
  Eigen::ArrayXXd deriv = Eigen::ArrayXXd::Zero(x.rows(), deriv_size);
  for (std::size_t i = 0; i < deriv_size; ++i) {
    for (auto const& dependency : param_dependencies[i]) {
      deriv.col(i) += reverse_eval[dependency];
    }
  }
  return std::make_pair(forward_eval.back(), deriv);
}

Eigen::ArrayXXd SimplifyAndEvaluate(const Eigen::ArrayX3i & stack,
                                    const Eigen::ArrayXXd & x,
                                    const Eigen::VectorXd &constants) {
  std::vector<bool> mask = FindUsedCommands(stack);
  std::vector<Eigen::ArrayXXd> forward_eval = EvaluateWithMask(stack, x, constants, mask);
  return forward_eval.back();
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> SimplifyAndEvaluateWithDerivative(
  const Eigen::ArrayX3i &stack,
  const Eigen::ArrayXXd &x,
  const Eigen::VectorXd &constants,
  const bool param_x_or_c) {
  std::vector<bool> mask = FindUsedCommands(stack);
  return EvaluateWithDerivativeAndMask(stack, x, constants, mask, param_x_or_c);
}

std::vector<bool> FindUsedCommands(const Eigen::ArrayX3i & stack) {
  std::vector<bool> used_command(stack.rows());
  used_command.back() = true;
  for (int i = stack.rows() - 1; i >= 0; --i) {
    if (used_command[i]) {
      for (int j = 0; j < get_arity(stack(i, 0)); ++j) {
        used_command[stack(i, j + 1)] = true;
      }
    }
  }
  return used_command;
}

std::vector<Eigen::ArrayXXd> EvaluateWithMask(const Eigen::ArrayX3i &stack,
                                              const Eigen::ArrayXXd &x,
                                              const Eigen::VectorXd &constants,
                                              const std::vector<bool> &mask) {
  std::vector<Eigen::ArrayXXd> forward_eval(stack.rows());

  for (int i = 0; i < stack.rows(); ++i) {
    if (mask[i]) {
      int node = stack(i, NODE_IDX);
      int op1 = stack(i, OP_1);
      int op2 = stack(i, OP_2);
      forward_eval[i] = backendnodes::forward_eval_function(
        node, op1, op2, x, constants, forward_eval);
    }
  }
  return forward_eval;
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvaluateWithDerivativeAndMask(
    const Eigen::ArrayX3i &stack,
    const Eigen::ArrayXXd &x,
    const Eigen::VectorXd &constants,
    const std::vector<bool> &mask,
    const bool param_x_or_c) {

  if (!BACKENDNODES) {
    return agraph_nodes_deriv(stack, x, constants, mask, param_x_or_c);
  }

  std::vector<Eigen::ArrayXXd> forward_eval = EvaluateWithMask(
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


Eigen::ArrayXXd Evaluate(const Eigen::ArrayX3i & stack,
                         const Eigen::ArrayXXd &x,
                         const Eigen::VectorXd &constants) {
  std::vector<bool> use_all_mask(stack.rows());
  std::fill(use_all_mask.begin(), use_all_mask.end(), true);
  std::vector<Eigen::ArrayXXd> forward_eval = EvaluateWithMask(
    stack, x, constants, use_all_mask);
  return forward_eval.back();  
}

std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvaluateWithDerivative(
    const Eigen::ArrayX3i &stack,
    const Eigen::ArrayXXd &x,
    const Eigen::VectorXd &constants,
    const bool param_x_or_c) {
  std::vector<bool> use_all_mask(stack.rows());
  std::fill(use_all_mask.begin(), use_all_mask.end(), true);
  return EvaluateWithDerivativeAndMask(stack, x, constants, use_all_mask, param_x_or_c);
}

Eigen::ArrayX3i SimplifyStack(const Eigen::ArrayX3i & stack) {
  std::vector<bool> used_command = FindUsedCommands(stack);
  std::map<int, int> reduced_param_map;
  Eigen::ArrayX3i new_stack(used_command.size(), 3);

  for (int i = 0, j = 0; i < stack.rows(); ++i) {
    if (used_command[i]) {
      for (int k = 0; k < get_arity(new_stack(j, 0)); ++k) {
        new_stack(j, k + 1) = reduced_param_map[new_stack(j, k + 1)];
      }
      reduced_param_map[i] = j;
      ++j;
    }
  }
  return new_stack;
}

Eigen::ArrayXXd reverse_eval_with_mask(const std::pair<int, int> deriv_shape,
                                       const int deriv_wrt_node,
                                       const std::vector<Eigen::ArrayXXd> &forward_eval,
                                       const Eigen::ArrayX3i &stack,
                                       const std::vector<bool> &mask) {
  int num_samples = deriv_shape.first;
  int num_features = deriv_shape.second;
  int stack_depth = stack.rows();
  Eigen::ArrayXXd zero = Eigen::ArrayXd::Zero(num_samples);
  Eigen::ArrayXXd ones = Eigen::ArrayXd::Ones(num_samples);

  Eigen::ArrayXXd derivative = Eigen::ArrayXXd::Zero(num_samples, num_features);
  std::vector<Eigen::ArrayXXd> reverse_eval(stack_depth); 
  std::fill(reverse_eval.begin(), reverse_eval.end(), zero); 
  reverse_eval[stack_depth-1] = ones;

  for (int i = stack_depth - 1; i >= 0; i--) {
    if (mask[i]) {
      int node = stack(i, NODE_IDX);
      int param1 = stack(i, OP_1);
      int param2 = stack(i, OP_2);
      if (node == deriv_wrt_node)  {
        derivative.col(param1) += reverse_eval[i];
      } else {
        backendnodes::reverse_eval_function(node, i, param1, param2,
                                            forward_eval,
                                            reverse_eval);
      }
    }
  }
  return derivative;
}
// void PrintStack(const Eigen::ArrayX3i & stack) {
//   // Prints a stack to std::cout.
//   for (std::size_t i = 0; i < stack.rows(); ++i) {
//     //this is the operator
//     std::cout << "(" << i << ") = " <<
//               oper_interface.operator_map[stack(i, 0)]->get_print() << " : ";

//     // Hard code for those with arity == 0
//     if (get_arity(stack(i, 0)]->get_arity() == 0) {
//       std::cout << " (" << stack(i, 1) << ")";
//     }

//     // loop through the rest dependent on their arity
//     for (int j = 0; j < get_arity(stack(i, 0)); ++j) {
//       std::cout << " (" << stack(i, j) << ")";
//     }

//     std::cout << std::endl;
//   }
// }