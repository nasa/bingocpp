/*!
 * \file acyclic_graph_nodes.cc
 *
 * \author Ethan Adams
 * \date 2/6/2018
 *
 * This file contains the functions associated with each class
 * implementation of Operation.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <Eigen/Dense>

#include <set>
#include <map>
#include <iostream>
#include <string>

#include "BingoCpp/acyclic_graph_nodes.h"

// X_Load
X_Load::X_Load() {}

Eigen::ArrayXXd X_Load::evaluate(const std::vector<int> &command,
                                 const Eigen::ArrayXXd &x,
                                 const std::vector<double> &constants,
                                 std::vector<Eigen::ArrayXXd> &buffer) {
  return x.col(command[0]);
}

void X_Load::deriv_evaluate(const std::vector<int> &command,
                            const int command_index,
                            const std::vector<Eigen::ArrayXXd> &forward_buffer,
                            std::vector<Eigen::ArrayXXd> &reverse_buffer,
                            int dependency) {
}

// C_Load
C_Load::C_Load() {}

Eigen::ArrayXXd C_Load::evaluate(const std::vector<int> &command,
                                 const Eigen::ArrayXXd &x,
                                 const std::vector<double> &constants,
                                 std::vector<Eigen::ArrayXXd> &buffer) {
  if (command[0] != -1) {
    return Eigen::ArrayXXd::Constant(x.rows(), 1,
                                     constants[command[0]]);

  } else {
    return Eigen::ArrayXXd::Zero(x.rows(), 1);
  }
}

void C_Load::deriv_evaluate(const std::vector<int> &command,
                            const int command_index,
                            const std::vector<Eigen::ArrayXXd> &forward_buffer,
                            std::vector<Eigen::ArrayXXd> &reverse_buffer,
                            int dependency) {
}

// Addition
Addition::Addition() {}

Eigen::ArrayXXd Addition::evaluate(const std::vector<int> &command,
                                   const Eigen::ArrayXXd &x,
                                   const std::vector<double> &constants,
                                   std::vector<Eigen::ArrayXXd> &buffer) {
  return buffer[command[0]] + buffer[command[1]];
}

void Addition::deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency) {
  if (command[0] == command_index) {
    reverse_buffer[command_index] += reverse_buffer[dependency];
  }

  if (command[1] == command_index) {
    reverse_buffer[command_index] += reverse_buffer[dependency];
  }
}

// Subtraction
Subtraction::Subtraction() {}

Eigen::ArrayXXd Subtraction::evaluate(const std::vector<int> &command,
                                      const Eigen::ArrayXXd &x,
                                      const std::vector<double> &constants,
                                      std::vector<Eigen::ArrayXXd> &buffer) {
  return buffer[command[0]] - buffer[command[1]];
}

void Subtraction::deriv_evaluate(const std::vector<int> &command,
                                 const int command_index,
                                 const std::vector<Eigen::ArrayXXd> &forward_buffer,
                                 std::vector<Eigen::ArrayXXd> &reverse_buffer,
                                 int dependency) {
  if (command[0] == command_index) {
    reverse_buffer[command_index] += reverse_buffer[dependency];
  }

  if (command[1] == command_index) {
    reverse_buffer[command_index] -= reverse_buffer[dependency];
  }
}

// Multiplication
Multiplication::Multiplication() {}

Eigen::ArrayXXd Multiplication::evaluate(const std::vector<int> &command,
    const Eigen::ArrayXXd &x,
    const std::vector<double> &constants,
    std::vector<Eigen::ArrayXXd> &buffer) {
  return buffer[command[0]] * buffer[command[1]];
}

void Multiplication::deriv_evaluate(const std::vector<int> &command,
                                    const int command_index,
                                    const std::vector<Eigen::ArrayXXd> &forward_buffer,
                                    std::vector<Eigen::ArrayXXd> &reverse_buffer,
                                    int dependency) {
  if (command[0] == command_index) {
    reverse_buffer[command_index] += reverse_buffer[dependency] *
                                     forward_buffer[command[1]];
  }

  if (command[1] == command_index) {
    reverse_buffer[command_index] += reverse_buffer[dependency] *
                                     forward_buffer[command[0]];
  }
}

// Division
Division::Division() {}

Eigen::ArrayXXd Division::evaluate(const std::vector<int> &command,
                                   const Eigen::ArrayXXd &x,
                                   const std::vector<double> &constants,
                                   std::vector<Eigen::ArrayXXd> &buffer) {
  return buffer[command[0]] / buffer[command[1]];
}

void Division::deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency) {
  if (command[0] == command_index) {
    reverse_buffer[command_index] += reverse_buffer[dependency] /
                                     forward_buffer[command[1]];
  }

  if (command[1] == command_index) {
    reverse_buffer[command_index] += reverse_buffer[dependency] *
                                     (-forward_buffer[dependency] /
                                      forward_buffer[command[1]]);
  }
}

std::map<int, Operation*> OperatorInterface::create_op_map() {
  std::map<int, Operation*> return_map;
  return_map[0] = new X_Load();
  return_map[1] = new C_Load();
  return_map[2] = new Addition();
  return_map[3] = new Subtraction();
  return_map[4] = new Multiplication();
  return_map[5] = new Division();
  return return_map;
}

std::map<int, Operation*> OperatorInterface::operator_map =
  OperatorInterface::create_op_map();
