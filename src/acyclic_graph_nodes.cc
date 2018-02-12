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

void X_Load::evaluate(const std::vector<int> &command,
                      const Eigen::ArrayXXd &x,
                      const std::vector<double> &constants,
                      std::vector<Eigen::ArrayXXd> &buffer,
                      std::size_t result_location) {
  buffer[result_location] = x.col(command[0]);
}

void X_Load::deriv_evaluate(const std::vector<int> &command,
                            const int command_index,
                            const std::vector<Eigen::ArrayXXd> &forward_buffer,
                            std::vector<Eigen::ArrayXXd> &reverse_buffer,
                            int dependency) {
}

// C_Load
C_Load::C_Load() {}

void C_Load::evaluate(const std::vector<int> &command,
                      const Eigen::ArrayXXd &x,
                      const std::vector<double> &constants,
                      std::vector<Eigen::ArrayXXd> &buffer,
                      std::size_t result_location) {
  if (command[0] != -1) {
    buffer[result_location] = Eigen::ArrayXXd::Constant(x.rows(), 1,
                              constants[command[0]]);

  } else {
    buffer[result_location] = Eigen::ArrayXXd::Zero(x.rows(), 1);
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

void Addition::evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location) {
  buffer[result_location] = buffer[command[0]] + buffer[command[1]];
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

void Subtraction::evaluate(const std::vector<int> &command,
                           const Eigen::ArrayXXd &x,
                           const std::vector<double> &constants,
                           std::vector<Eigen::ArrayXXd> &buffer,
                           std::size_t result_location) {
  buffer[result_location] = buffer[command[0]] - buffer[command[1]];
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

void Multiplication::evaluate(const std::vector<int> &command,
                              const Eigen::ArrayXXd &x,
                              const std::vector<double> &constants,
                              std::vector<Eigen::ArrayXXd> &buffer,
                              std::size_t result_location) {
  buffer[result_location] = buffer[command[0]] * buffer[command[1]];
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

void Division::evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location) {
  buffer[result_location] = buffer[command[0]] / buffer[command[1]];
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

// Sin
Sin::Sin() {}

void Sin::evaluate(const std::vector<int> &command,
                   const Eigen::ArrayXXd &x,
                   const std::vector<double> &constants,
                   std::vector<Eigen::ArrayXXd> &buffer,
                   std::size_t result_location) {
  buffer[result_location] = buffer[command[0]].sin();
}

void Sin::deriv_evaluate(const std::vector<int> &command,
                         const int command_index,
                         const std::vector<Eigen::ArrayXXd> &forward_buffer,
                         std::vector<Eigen::ArrayXXd> &reverse_buffer,
                         int dependency) {
  // TODO deriv of sin
  reverse_buffer[command_index] += reverse_buffer[dependency].cos();
}

// Cos
Cos::Cos() {}

void Cos::evaluate(const std::vector<int> &command,
                   const Eigen::ArrayXXd &x,
                   const std::vector<double> &constants,
                   std::vector<Eigen::ArrayXXd> &buffer,
                   std::size_t result_location) {
  buffer[result_location] = buffer[command[0]].sin();
}

void Cos::deriv_evaluate(const std::vector<int> &command,
                         const int command_index,
                         const std::vector<Eigen::ArrayXXd> &forward_buffer,
                         std::vector<Eigen::ArrayXXd> &reverse_buffer,
                         int dependency) {
  // TODO deriv of cos
  reverse_buffer[command_index] += reverse_buffer[dependency].sin();
}

// Exp
Exp::Exp() {}

void Exp::evaluate(const std::vector<int> &command,
                   const Eigen::ArrayXXd &x,
                   const std::vector<double> &constants,
                   std::vector<Eigen::ArrayXXd> &buffer,
                   std::size_t result_location) {
  buffer[result_location] = buffer[command[0]].exp();
}

void Exp::deriv_evaluate(const std::vector<int> &command,
                         const int command_index,
                         const std::vector<Eigen::ArrayXXd> &forward_buffer,
                         std::vector<Eigen::ArrayXXd> &reverse_buffer,
                         int dependency) {
  // TODO deriv of exp
}

// Log
Log::Log() {}

void Log::evaluate(const std::vector<int> &command,
                   const Eigen::ArrayXXd &x,
                   const std::vector<double> &constants,
                   std::vector<Eigen::ArrayXXd> &buffer,
                   std::size_t result_location) {
  buffer[result_location] = (buffer[command[0]].abs()).log();
}

void Log::deriv_evaluate(const std::vector<int> &command,
                         const int command_index,
                         const std::vector<Eigen::ArrayXXd> &forward_buffer,
                         std::vector<Eigen::ArrayXXd> &reverse_buffer,
                         int dependency) {
  // TODO deriv of log
}

// Power
Power::Power() {}

void Power::evaluate(const std::vector<int> &command,
                     const Eigen::ArrayXXd &x,
                     const std::vector<double> &constants,
                     std::vector<Eigen::ArrayXXd> &buffer,
                     std::size_t result_location) {
  buffer[result_location] = buffer[command[0]].pow(buffer[command[1]]);
}

void Power::deriv_evaluate(const std::vector<int> &command,
                           const int command_index,
                           const std::vector<Eigen::ArrayXXd> &forward_buffer,
                           std::vector<Eigen::ArrayXXd> &reverse_buffer,
                           int dependency) {
  // TODO deriv of power
}

// Absolute
Absolute::Absolute() {}

void Absolute::evaluate(const std::vector<int> &command,
                        const Eigen::ArrayXXd &x,
                        const std::vector<double> &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location) {
  buffer[result_location] = buffer[command[0]].abs();
}

void Absolute::deriv_evaluate(const std::vector<int> &command,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency) {
  // TODO deriv of absolute
}

// Sqrt
Sqrt::Sqrt() {}

void Sqrt::evaluate(const std::vector<int> &command,
                    const Eigen::ArrayXXd &x,
                    const std::vector<double> &constants,
                    std::vector<Eigen::ArrayXXd> &buffer,
                    std::size_t result_location) {
  buffer[result_location] = (buffer[command[0]].abs()).sqrt();
}

void Sqrt::deriv_evaluate(const std::vector<int> &command,
                          const int command_index,
                          const std::vector<Eigen::ArrayXXd> &forward_buffer,
                          std::vector<Eigen::ArrayXXd> &reverse_buffer,
                          int dependency) {
  // TODO deriv of square root
}

std::map<int, Operation*> OperatorInterface::create_op_map() {
  std::map<int, Operation*> return_map;
  return_map[0] = new X_Load();
  return_map[1] = new C_Load();
  return_map[2] = new Addition();
  return_map[3] = new Subtraction();
  return_map[4] = new Multiplication();
  return_map[5] = new Division();
  return_map[6] = new Sin();
  return_map[7] = new Cos();
  return_map[8] = new Exp();
  return_map[9] = new Log();
  return_map[10] = new Power();
  return_map[11] = new Absolute();
  return_map[12] = new Sqrt();
  return return_map;
}

std::map<int, Operation*> OperatorInterface::operator_map =
  OperatorInterface::create_op_map();
