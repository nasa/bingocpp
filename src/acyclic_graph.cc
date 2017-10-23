// A simple program that computes the square root of a number
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <Eigen/Dense>

#include <map>
#include <iostream>

#include "BingoCpp/acyclic_graph.hh"



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
  CommandStack simple_stack = SimplifyStack(stack);
  return Evaluate(simple_stack, x, constants);
}

Eigen::ArrayXXd Evaluate(CommandStack stack, Eigen::ArrayXXd x,
                         std::vector<double> constants) {
  std::vector<Eigen::ArrayXXd> R(stack.size());

  for (std::size_t i = 0; i < stack.size(); ++i) {
    switch (stack[i].first) {
      case 0:
        R[i] = x.col(stack[i].second[0]);
        break;

      case 1:
        R[i] = Eigen::ArrayXXd::Constant(x.rows(), 1,
                                         constants[stack[i].second[0]]);
        break;

      case 2:
        R[i] = R[stack[i].second[0]] + R[stack[i].second[1]];
        break;

      case 3:
        R[i] = R[stack[i].second[0]] - R[stack[i].second[1]];
        break;

      case 4:
        R[i] = R[stack[i].second[0]] * R[stack[i].second[1]];
        break;

      case 5:
        R[i] = R[stack[i].second[0]] / R[stack[i].second[1]];
        break;

      default:
        break;
    }

    // std::cout << "R["<< i <<"] = " << R[i] << "\n\n";
  }

  return R.back();
}



void PrintStack(CommandStack stack) {
  for (auto const& command : stack) {
    std::cout << OperatorString[command.first] << " : ";

    for (auto const& param : command.second) {
      std::cout << " (" << param << ")";
    }

    std::cout << std::endl;
  }
}



CommandStack SimplifyStack(CommandStack stack) {
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







