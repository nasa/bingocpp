/*!
 * \file acyclic_graph.cc
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This file contains the functions associated with an acyclic graph
 * representation of a symbolic equation.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <Eigen/Dense>

#include <set>
#include <map>
#include <iostream>

#include "BingoCpp/acyclic_graph.h"
#include "BingoCpp/acyclic_graph_nodes.h"
#include "BingoCpp/backend_nodes.h"

// create an instance of the Operator_Interface map
OperatorInterface oper_interface;

bool IsCpp() {
    return true;
}


void ReverseSingleCommand(const Eigen::ArrayX3i &stack,
                          const int command_index,
                          const std::vector<Eigen::ArrayXXd> &forward_buffer,
                          std::vector<Eigen::ArrayXXd> &reverse_buffer,
                          const std::set<int> &dependencies) {
  // Computes reverse autodiff partial of a stack command.
  for (auto const& dependency : dependencies) {
    oper_interface.operator_map[stack(dependency, 0)]->deriv_evaluate(
      stack, command_index, forward_buffer, reverse_buffer,
      dependency);
  }
}



Eigen::ArrayXXd Evaluate(const Eigen::ArrayX3i & stack,
                         const Eigen::ArrayXXd &x,
                         const Eigen::VectorXd &constants) {
  // Evaluates a stack at the given x using the given constants.
  std::vector<Eigen::ArrayXXd> forward_eval(stack.rows());

  for (std::size_t i = 0; i < stack.rows(); ++i) {
    oper_interface.operator_map[stack(i, 0)]->evaluate(
      stack, x, constants, forward_eval, i);
  }

  return forward_eval.back();
}



Eigen::ArrayXXd SimplifyAndEvaluate(const Eigen::ArrayX3i & stack,
                                    const Eigen::ArrayXXd & x,
                                    const Eigen::VectorXd &constants) {
  std::vector<bool> mask = FindUsedCommands(stack);
  return EvaluateWithMask(stack, x, constants, mask);
}



Eigen::ArrayXXd EvaluateWithMask(const Eigen::ArrayX3i &stack,
                                 const Eigen::ArrayXXd &x,
                                 const Eigen::VectorXd &constants,
                                 const std::vector<bool> &mask) {
  Eigen::ArrayXXd forward_eval = Eigen::ArrayXXd(stack.rows(), x.rows());
  for (std::size_t i = 0; i < stack.rows(); ++i) {
    if (mask[i]) {
      int node = stack(i, 0);
      int param1 = stack(i, 1);
      int param2 = stack(i, 2);
      forward_eval.row(i) = backendnodes::forward_eval_function(
        node, param1, param2, x, constants, forward_eval);
    }
  }
  return (forward_eval.row(forward_eval.rows() - 1)).transpose();
}



std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvaluateWithDerivative(
  const Eigen::ArrayX3i &stack,
  const Eigen::ArrayXXd &x,
  const Eigen::VectorXd &constants,
  const bool param_x_or_c) {
  // Evaluates a stack and its derivative with the given x and constants.
  std::vector<Eigen::ArrayXXd> forward_eval(stack.rows());
  std::vector<std::set<int>> stack_dependencies(stack.rows(), std::set<int>());
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
    oper_interface.operator_map[stack(i, 0)]->evaluate(
      stack, x, constants, forward_eval, i);

    if (stack(i, 0) == deriv_operator_number) {
      param_dependencies[stack(i, 1)].insert(i);
    }

    for (int j = 0; j < oper_interface.operator_map[stack(i, 0)]->get_arity();
         ++j) {
      stack_dependencies[stack(i, j + 1)].insert(i);
    }
  }

  // reverse pass through stack
  std::vector<Eigen::ArrayXXd> reverse_eval(stack.rows(),
      Eigen::ArrayXXd::Zero(x.rows(), 1));
  reverse_eval[stack.rows() - 1] = Eigen::ArrayXXd::Ones(x.rows(), 1);

  for (int i = stack.rows() - 2; i >= 0; --i) {
    ReverseSingleCommand(stack, i, forward_eval, reverse_eval,
                         stack_dependencies[i]);
  }

  // build derivative array
  Eigen::ArrayXXd deriv = Eigen::ArrayXXd::Zero(x.rows(), deriv_size);

  for (std::size_t i = 0; i < x.cols(); ++i) {
    for (auto const& dependency : param_dependencies[i]) {
      deriv.col(i) += reverse_eval[dependency];
    }
  }

  return std::make_pair(forward_eval.back(), deriv);
}



std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> SimplifyAndEvaluateWithDerivative(
  const Eigen::ArrayX3i &stack,
  const Eigen::ArrayXXd &x,
  const Eigen::VectorXd &constants,
  const bool param_x_or_c) {
  // Evaluates a stack and its derivative, but only the utilized commands.
  std::vector<bool> mask = FindUsedCommands(stack);
  return EvaluateWithDerivativeAndMask(stack, x, constants, mask, param_x_or_c);
}



std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvaluateWithDerivativeAndMask(
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
      oper_interface.operator_map[stack(i, 0)]->evaluate(
        stack, x, constants, forward_eval, i);
      if (stack(i, 0) == deriv_operator_number) {
        param_dependencies[stack(i, 1)].insert(i);
      }
      for (int j = 0; j < oper_interface.operator_map[stack(i, 0)]->get_arity();
           ++j) {
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



void PrintStack(const Eigen::ArrayX3i & stack) {
  // Prints a stack to std::cout.
  for (std::size_t i = 0; i < stack.rows(); ++i) {
    //this is the operator
    std::cout << "(" << i << ") = " <<
              oper_interface.operator_map[stack(i, 0)]->get_print() << " : ";

    // Hard code for those with arity == 0
    if (oper_interface.operator_map[stack(i, 0)]->get_arity() == 0) {
      std::cout << " (" << stack(i, 1) << ")";
    }

    // loop through the rest dependent on their arity
    for (int j = 0; j < oper_interface.operator_map[stack(i, 0)]->get_arity();
         ++j) {
      std::cout << " (" << stack(i, j) << ")";
    }

    std::cout << std::endl;
  }
}



Eigen::ArrayX3i SimplifyStack(const Eigen::ArrayX3i & stack) {
  // Simplifies a stack.
  std::vector<bool> used_command = FindUsedCommands(stack);
  std::map<int, int> reduced_param_map;
  Eigen::ArrayX3i new_stack(used_command.size(), 3);

  // TODO(gbomarito)  would size_t be faster?
  for (int i = 0, j = 0; i < stack.rows(); ++i) {
    if (used_command[i]) {
      for (std::size_t k = 0; k < oper_interface.operator_map[new_stack(j, 0)]
           ->get_arity(); ++k) {
        new_stack(j, k + 1) = reduced_param_map[new_stack(j, k + 1)];
      }

      reduced_param_map[i] = j;
      ++j;
    }
  }

  return new_stack;
}



std::vector<bool> FindUsedCommands(const Eigen::ArrayX3i & stack) {
  // Finds which commands are utilized in a stack.
  std::vector<bool> used_command(stack.rows());
  used_command.back() = true;

  for (int i = stack.rows() - 1; i >= 0; --i) {
    if (used_command[i]) {
      for (std::size_t j = 0; j < oper_interface.operator_map[stack(i, 0)]
           ->get_arity(); ++j) {
        used_command[stack(i, j + 1)] = true;
      }
    }
  }

  return used_command;
}
