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

// create an instance of the Operator_Interface map
OperatorInterface oper_interface;

void ForwardSingleCommand(const SingleCommand &command,
                          const Eigen::ArrayXXd &x,
                          const std::vector<double> &constants,
                          std::vector<Eigen::ArrayXXd> &buffer,
                          std::size_t result_location) {
  oper_interface.operator_map[command.first]->evaluate(
    command.second, x, constants, buffer, result_location);
}



void ReverseSingleCommand(const CommandStack &stack,
                          const int command_index,
                          const std::vector<Eigen::ArrayXXd> &forward_buffer,
                          std::vector<Eigen::ArrayXXd> &reverse_buffer,
                          const std::set<int> &dependencies) {
  // Computes reverse autodiff partial of a stack command.
  for (auto const& dependency : dependencies) {
    oper_interface.operator_map[stack[dependency].first]->deriv_evaluate(
      stack[dependency].second, command_index, forward_buffer, reverse_buffer,
      dependency);
  }
}



Eigen::ArrayXXd Evaluate(const CommandStack & stack,
                         const Eigen::ArrayXXd &x,
                         const std::vector<double> &constants) {
  // Evaluates a stack at the given x using the given constants.
  std::vector<Eigen::ArrayXXd> forward_eval(stack.size());

  for (std::size_t i = 0; i < stack.size(); ++i) {
    ForwardSingleCommand(stack[i], x, constants, forward_eval, i);
  }

  return forward_eval.back();
}



Eigen::ArrayXXd SimplifyAndEvaluate(const CommandStack & stack,
                                    const Eigen::ArrayXXd & x,
                                    const std::vector<double> &constants) {
  // Evaluates a stack, but only the commands that are utilized.
  std::vector<bool> mask = FindUsedCommands(stack);
  return EvaluateWithMask(stack, x, constants, mask);
}



Eigen::ArrayXXd EvaluateWithMask(const CommandStack & stack,
                                 const Eigen::ArrayXXd & x,
                                 const std::vector<double> &constants,
                                 const std::vector<bool> &mask) {
  // Evaluates a stack at the given x using the given constants.
  std::vector<Eigen::ArrayXXd> forward_eval(stack.size());

  for (std::size_t i = 0; i < stack.size(); ++i) {
    if (mask[i]) {
      ForwardSingleCommand(stack[i], x, constants, forward_eval, i);
    }
  }

  return forward_eval.back();
}



std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvaluateWithDerivative(
  const CommandStack &stack,
  const Eigen::ArrayXXd &x,
  const std::vector<double> &constants) {
  // Evaluates a stack and its derivative with the given x and constants.
  std::vector<Eigen::ArrayXXd> forward_eval(stack.size());
  std::vector<std::set<int>> x_dependencies(x.cols(), std::set<int>());
  std::vector<std::set<int>> stack_dependencies(stack.size(), std::set<int>());

  // forward eval with dependencies
  for (std::size_t i = 0; i < stack.size(); ++i) {
    ForwardSingleCommand(stack[i], x, constants, forward_eval, i);

    // TODO(gbomarito) thish could be more general based on arity, etc
    if (stack[i].first == 0) {
      x_dependencies[stack[i].second[0]].insert(i);
    }

    for (int j = 0; j < oper_interface.operator_map[stack[i].first]->get_arity();
         ++j) {
      stack_dependencies[stack[i].second[j]].insert(i);
    }
  }

  // reverse pass through stack
  std::vector<Eigen::ArrayXXd> reverse_eval(stack.size(),
      Eigen::ArrayXXd::Zero(x.rows(), 1));
  reverse_eval[stack.size() - 1] = Eigen::ArrayXXd::Ones(x.rows(), 1);

  for (int i = stack.size() - 2; i >= 0; --i) {
    ReverseSingleCommand(stack, i, forward_eval, reverse_eval,
                         stack_dependencies[i]);
  }

  // build derivative array
  Eigen::ArrayXXd deriv_x = Eigen::ArrayXXd::Zero(x.rows(), x.cols());

  for (std::size_t i = 0; i < x.cols(); ++i) {
    for (auto const& dependency : x_dependencies[i]) {
      deriv_x.col(i) += reverse_eval[dependency];
    }
  }

  return std::make_pair(forward_eval.back(), deriv_x);
}



std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> SimplifyAndEvaluateWithDerivative(
  const CommandStack &stack,
  const Eigen::ArrayXXd &x,
  const std::vector<double> &constants) {
  // Evaluates a stack and its derivative, but only the utilized commands.
  std::vector<bool> mask = FindUsedCommands(stack);
  return EvaluateWithDerivativeAndMask(stack, x, constants, mask);
}



std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvaluateWithDerivativeAndMask(
  const CommandStack &stack,
  const Eigen::ArrayXXd &x,
  const std::vector<double> &constants,
  const std::vector<bool> &mask) {
  // Evaluates a stack and its derivative with the given x and constants.
  std::vector<Eigen::ArrayXXd> forward_eval(stack.size());
  std::vector<std::set<int>> x_dependencies(x.cols(), std::set<int>());
  std::vector<std::set<int>> stack_dependencies(stack.size(),
                          std::set<int>());

  // forward eval with dependencies
  for (std::size_t i = 0; i < stack.size(); ++i) {
    if (mask[i]) {
      ForwardSingleCommand(stack[i], x, constants, forward_eval, i);

      // TODO(gbomarito) thish could be more general based on arity, etc
      if (stack[i].first == 0) {
        x_dependencies[stack[i].second[0]].insert(i);
      }

      for (int j = 0; j < oper_interface.operator_map[stack[i].first]->get_arity();
           ++j) {
        stack_dependencies[stack[i].second[j]].insert(i);
      }
    }
  }

  // reverse pass through stack
  std::vector<Eigen::ArrayXXd> reverse_eval(stack.size());
  reverse_eval[stack.size() - 1] = Eigen::ArrayXXd::Ones(x.rows(), 1);

  for (int i = stack.size() - 2; i >= 0; --i) {
    if (mask[i]) {
      reverse_eval[i] = Eigen::ArrayXXd::Zero(x.rows(), 1);
      ReverseSingleCommand(stack, i, forward_eval, reverse_eval,
                           stack_dependencies[i]);
    }
  }

  // build derivative array
  Eigen::ArrayXXd deriv_x = Eigen::ArrayXXd::Zero(x.rows(), x.cols());

  for (std::size_t i = 0; i < x.cols(); ++i) {
    for (auto const& dependency : x_dependencies[i]) {
      deriv_x.col(i) += reverse_eval[dependency];
    }
  }

  return std::make_pair(forward_eval.back(), deriv_x);
}



void PrintStack(const CommandStack & stack) {
  // Prints a stack to std::cout.
  for (std::size_t i = 0; i < stack.size(); ++i) {
    //this is the operator
    std::cout << "(" << i << ") = " <<
              oper_interface.operator_map[stack[i].first]->get_print() << " : ";

    for (auto const& param : stack[i].second) {
      std::cout << " (" << param << ")";
    }

    std::cout << std::endl;
  }
}



CommandStack SimplifyStack(const CommandStack & stack) {
  // Simplifies a stack.
  std::vector<bool> used_command = FindUsedCommands(stack);
  std::map<int, int> reduced_param_map;
  CommandStack new_stack;

  // TODO(gbomarito)  would size_t be faster?
  for (int i = 0, j = 0; i < stack.size(); ++i) {
    if (used_command[i]) {
      new_stack.push_back(stack[i]);

      // TODO(gbomarito) should look up whether node is terminal or not
      if (oper_interface.operator_map[new_stack[j].first]->get_arity() > 0) {
        for (std::size_t k = 0; k < new_stack[j].second.size(); ++k) {
          new_stack[j].second[k] = reduced_param_map[new_stack[j].second[k]];
        }
      }

      reduced_param_map[i] = j;
      ++j;
    }
  }

  return new_stack;
}



std::vector<bool> FindUsedCommands(const CommandStack & stack) {
  // Finds which commands are utilized in a stack.
  std::vector<bool> used_command(stack.size());
  used_command.back() = true;

  for (int i = stack.size() - 1; i >= 0; --i) {
    if (used_command[i]) {
      for (std::size_t j = 0; j < oper_interface.operator_map[stack[i].first]
                                  ->get_arity(); ++j) {
          used_command[stack[i].second[j]] = true;
      }
    }
  }

  return used_command;
}





