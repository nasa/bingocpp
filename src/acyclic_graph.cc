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

#include <map>
#include <iostream>

#include "BingoCpp/acyclic_graph.h"


// The print strings accociated with each function
std::map<int, std::string> OperatorString = {
  {0, "X"},
  {1, "C"},
  {2, "+"},
  {3, "-"},
  {4, "*"},
  {5, "/"}
};


Eigen::ArrayXXd SimplifyAndEvaluate(CommandStack stack, Eigen::ArrayXXd x,
                                    std::vector<double> constants) {
  // Simplifies a stack then evaluates it.
  CommandStack simple_stack = SimplifyStack(stack);
  return Evaluate(simple_stack, x, constants);
}


std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> SimplifyAndEvaluateWithDerivative(
  CommandStack stack, Eigen::ArrayXXd x, std::vector<double> constants) {
  // Evaluates a stack and its derivative after simplification.
  CommandStack simple_stack = SimplifyStack(stack);
  return EvaluateWithDerivative(simple_stack, x, constants);
}


Eigen::ArrayXXd Evaluate(CommandStack stack, Eigen::ArrayXXd x,
                         std::vector<double> constants) {
  // Evaluates a stack at the given x using the given constants.
  std::vector<Eigen::ArrayXXd> forward_eval(stack.size());

  for (std::size_t i = 0; i < stack.size(); ++i) {
    switch (stack[i].first) {
      case 0:
        forward_eval[i] = x.col(stack[i].second[0]);
        break;

      case 1:
        forward_eval[i] = Eigen::ArrayXXd::Constant(x.rows(), 1,
                          constants[stack[i].second[0]]);
        break;

      case 2:
        forward_eval[i] = forward_eval[stack[i].second[0]] +
                          forward_eval[stack[i].second[1]];
        break;

      case 3:
        forward_eval[i] = forward_eval[stack[i].second[0]] -
                          forward_eval[stack[i].second[1]];
        break;

      case 4:
        forward_eval[i] = forward_eval[stack[i].second[0]] *
                          forward_eval[stack[i].second[1]];
        break;

      case 5:
        forward_eval[i] = forward_eval[stack[i].second[0]] /
                          forward_eval[stack[i].second[1]];
        break;

      default:
        break;
    }

    // std::cout << "R["<< i <<"] = " << R[i] << "\n\n";
  }

  return forward_eval.back();
}



std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvaluateWithDerivative(
  CommandStack stack, Eigen::ArrayXXd x, std::vector<double> constants) {
  // Evaluates a stack and its derivative with the given x and constants.
  std::vector<Eigen::ArrayXXd> forward_eval(stack.size());
  std::vector<Eigen::ArrayXXd> reverse_eval(stack.size());
  std::vector<std::vector<int>> x_dependencies(x.cols(), std::vector<int>(0));
  std::vector<std::vector<int>> stack_dependencies(stack.size(),
                             std::vector<int>(0));

  // TODO(gbomarito) refactor this function
  // forward eval with dependencies
  for (std::size_t i = 0; i < stack.size(); ++i) {
    switch (stack[i].first) {
      case 0:
        forward_eval[i] = x.col(stack[i].second[0]);
        x_dependencies[stack[i].second[0]].push_back(i);
        break;

      case 1:
        forward_eval[i] = Eigen::ArrayXXd::Constant(x.rows(), 1,
                          constants[stack[i].second[0]]);
        break;

      case 2:
        forward_eval[i] = forward_eval[stack[i].second[0]] +
                          forward_eval[stack[i].second[1]];
        stack_dependencies[stack[i].second[0]].push_back(i);
        stack_dependencies[stack[i].second[1]].push_back(i);
        break;

      case 3:
        forward_eval[i] = forward_eval[stack[i].second[0]] -
                          forward_eval[stack[i].second[1]];
        stack_dependencies[stack[i].second[0]].push_back(i);
        stack_dependencies[stack[i].second[1]].push_back(i);
        break;

      case 4:
        forward_eval[i] = forward_eval[stack[i].second[0]] *
                          forward_eval[stack[i].second[1]];
        stack_dependencies[stack[i].second[0]].push_back(i);
        stack_dependencies[stack[i].second[1]].push_back(i);
        break;

      case 5:
        forward_eval[i] = forward_eval[stack[i].second[0]] /
                          forward_eval[stack[i].second[1]];
        stack_dependencies[stack[i].second[0]].push_back(i);
        stack_dependencies[stack[i].second[1]].push_back(i);
        break;

      default:
        break;
    }
  }

  // reverse pass through stack
  reverse_eval[stack.size() - 1] = Eigen::ArrayXXd::Ones(x.rows(), 1);

  for (int i = stack.size() - 2; i >= 0; --i) {
    reverse_eval[i] = Eigen::ArrayXXd::Zero(x.rows(), 1);

    for (auto const& dependency : stack_dependencies[i]) {
      switch (stack[dependency].first) {
        case 2:  // + add
          reverse_eval[i] += reverse_eval[dependency];
          break;

        case 3:  // - subtract
          if (stack[dependency].second[0] == i) {
            reverse_eval[i] += reverse_eval[dependency];

          } else {
            reverse_eval[i] -= reverse_eval[dependency];
          }

          break;

        case 4:  // * multiply
          if (stack[dependency].second[0] == i) {
            reverse_eval[i] += reverse_eval[dependency] *
                               forward_eval[stack[dependency].second[1]];

          } else {
            reverse_eval[i] += reverse_eval[dependency] *
                               forward_eval[stack[dependency].second[0]];
          }

          break;

        case 5:  // / divide
          if (stack[dependency].second[0] == i) {
            reverse_eval[i] += reverse_eval[dependency] /
                               forward_eval[stack[dependency].second[1]];

          } else {
            reverse_eval[i] += reverse_eval[dependency] *
                               (-forward_eval[dependency] /
                                 forward_eval[stack[dependency].second[1]]);
          }

          break;

        default:
          break;
      }
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



void PrintStack(CommandStack stack) {
  // Prints a stack to std::cout.
  for (std::size_t i = 0; i < stack.size(); ++i) {
    std::cout << "(" << i << ") = " << OperatorString[stack[i].first] << " : ";

    for (auto const& param : stack[i].second) {
      std::cout << " (" << param << ")";
    }

    std::cout << std::endl;
  }
}



CommandStack SimplifyStack(CommandStack stack) {
  // Simplifies a stack.
  std::vector<bool> used_command = FindUsedCommands(stack);
  std::map<int, int> reduced_param_map;
  CommandStack new_stack;

  // TODO(gbomarito)  would size_t be faster?
  for (int i = 0, j = 0; i < stack.size(); ++i) {
    if (used_command[i]) {
      new_stack.push_back(stack[i]);

      // TODO(gbomarito) should look up whether node is terminal or not
      if (new_stack[j].first > 1) {
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



std::vector<bool> FindUsedCommands(CommandStack stack) {
  // Finds which commands are utilized in a stack.
  std::vector<bool> used_command(stack.size());
  used_command.back() = true;

  for (int i = stack.size() - 1; i >= 0; --i) {
    if (used_command[i]) {
      // TODO(gbomarito) should look up whether node is terminal or not
      if (stack[i].first > 1) {
        for (auto const& param : stack[i].second) {
          used_command[param] = true;
        }
      }
    }
  }

  return used_command;
}






