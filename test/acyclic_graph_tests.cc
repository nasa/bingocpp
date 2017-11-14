/*!
 * \file acyclic_graph_tests.cc
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This file contains the unit tests for the functions associated with the
 * acyclic graph representation of a symbolic equation.
 */

#include <stdio.h>
#include <math.h>

#include <iostream>

#include "gtest/gtest.h"
#include "BingoCpp/acyclic_graph.h"

namespace {

class AcyclicGraphTest : public::testing::Test {
 public:
  CommandStack stack;
  CommandStack stack2;
  Eigen::ArrayXXd x;
  std::vector<double> constants;

  AcyclicGraphTest(): stack(), x(3, 3), constants() {
  }

  void SetUp() {
    // y = x_0 * ( C_0 + C_1/x_1 ) - x_0
    stack.push_back(std::make_pair(0, std::vector<int>()));
    stack[0].second.push_back(0);
    stack.push_back(std::make_pair(0, std::vector<int>()));
    stack[1].second.push_back(1);
    stack.push_back(std::make_pair(1, std::vector<int>()));
    stack[2].second.push_back(0);
    stack.push_back(std::make_pair(1, std::vector<int>()));
    stack[3].second.push_back(1);
    stack.push_back(std::make_pair(5, std::vector<int>()));
    stack[4].second.push_back(3);
    stack[4].second.push_back(1);
    stack.push_back(std::make_pair(5, std::vector<int>()));
    stack[5].second.push_back(3);
    stack[5].second.push_back(1);
    stack.push_back(std::make_pair(2, std::vector<int>()));
    stack[6].second.push_back(4);
    stack[6].second.push_back(2);
    stack.push_back(std::make_pair(2, std::vector<int>()));
    stack[7].second.push_back(4);
    stack[7].second.push_back(2);
    stack.push_back(std::make_pair(4, std::vector<int>()));
    stack[8].second.push_back(6);
    stack[8].second.push_back(0);
    stack.push_back(std::make_pair(4, std::vector<int>()));
    stack[9].second.push_back(5);
    stack[9].second.push_back(6);
    stack.push_back(std::make_pair(3, std::vector<int>()));
    stack[10].second.push_back(7);
    stack[10].second.push_back(6);
    stack.push_back(std::make_pair(3, std::vector<int>()));
    stack[11].second.push_back(8);
    stack[11].second.push_back(0);
    // y = x_0 * x_0
    stack2.push_back(std::make_pair(0, std::vector<int>()));
    stack2[0].second.push_back(0);
    stack2.push_back(std::make_pair(4, std::vector<int>()));
    stack2[1].second.push_back(0);
    stack2[1].second.push_back(0);
    x << 1., 4., 7., 2., 5., 8., 3., 6., 9.;
    constants.push_back(3.14);
    constants.push_back(10.0);
  }

  void TearDown() {
    // code here will be called just after the test completes
  }
};







TEST_F(AcyclicGraphTest, evaluate) {
  Eigen::ArrayXXd y = Evaluate(stack, x, constants);
  Eigen::ArrayXXd y_true = x.col(0) * (constants[0] + constants[1] / x.col(1))
                           - x.col(0);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(y(i), y_true(i));
  }
}


TEST_F(AcyclicGraphTest, derivative) {
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> y_and_dy =
    EvaluateWithDerivative(stack, x, constants);
  Eigen::ArrayXXd y_true = x.col(0) * (constants[0] + constants[1] / x.col(1))
                           - x.col(0);
  Eigen::ArrayXXd dy_true = Eigen::ArrayXXd::Zero(3, 3);
  dy_true.col(0) = constants[0] + constants[1] / x.col(1) - 1.;
  dy_true.col(1) = - x.col(0) * constants[1] / x.col(1) / x.col(1);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(y_and_dy.first(i), y_true(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(y_and_dy.second(i), dy_true(i));
  }
}


TEST_F(AcyclicGraphTest, maskevaluate) {
  Eigen::ArrayXXd y = Evaluate(stack, x, constants);
  Eigen::ArrayXXd y_simple = SimplifyAndEvaluate(stack, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(y(i), y_simple(i));
  }
}


TEST_F(AcyclicGraphTest, maskderivative) {
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> y_and_dy =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> y_and_dy_simple =
    SimplifyAndEvaluateWithDerivative(stack, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(y_and_dy.first(i), y_and_dy_simple.first(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(y_and_dy.second(i), y_and_dy_simple.second(i));
  }
}


TEST_F(AcyclicGraphTest, simplify) {
  // shorter stack
  CommandStack short_stack = SimplifyStack(stack);
  ASSERT_LE(short_stack.size(), stack.size());
  // equivalent evatuation
  Eigen::ArrayXXd y = Evaluate(stack, x, constants);
  Eigen::ArrayXXd simplified_y = Evaluate(short_stack, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(y(i), simplified_y(i));
  }
}


TEST_F(AcyclicGraphTest, utilization) {
  std::vector<bool> used_commands = FindUsedCommands(stack);
  int num_used_commands = 0;

  for (auto const& command_is_used : used_commands) {
    if (command_is_used) {
      ++num_used_commands;
    }
  }

  ASSERT_EQ(num_used_commands, 8);
}


TEST_F(AcyclicGraphTest, squaredfunc) {
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> y_and_dy =
    EvaluateWithDerivative(stack2, x, constants);
  Eigen::ArrayXXd dy_true = Eigen::ArrayXXd::Zero(3, 3);
  dy_true.col(0) = 2. * x.col(0);

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(y_and_dy.second(i), dy_true(i));
  }
}


}  // namespace



