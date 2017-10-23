#include <stdio.h>
#include <math.h>

#include <iostream>

#include "gtest/gtest.h"
#include "BingoCpp/acyclic_graph.hh"

namespace {

class AcyclicGraphTest : public::testing::Test {
 public:
  CommandStack stack;
  Eigen::ArrayXXd x;
  std::vector<double> constants;

  AcyclicGraphTest(): stack(), x(3, 3), constants() {
  }

  void SetUp() {
    stack.push_back(std::make_pair(1, std::vector<int>()));
    stack[0].second.push_back(0);
    stack.push_back(std::make_pair(0, std::vector<int>()));
    stack[1].second.push_back(1);
    stack.push_back(std::make_pair(2, std::vector<int>()));
    stack[2].second.push_back(0);
    stack[2].second.push_back(1);
    stack.push_back(std::make_pair(2, std::vector<int>()));
    stack[3].second.push_back(1);
    stack[3].second.push_back(2);
    stack.push_back(std::make_pair(2, std::vector<int>()));
    stack[4].second.push_back(1);
    stack[4].second.push_back(2);
    x << 1., 4., 7., 2., 5., 8., 3., 6., 9.;
    constants.push_back(3.14);
  }

  void TearDown() {
    // code here will be called just after the test completes
  }
};







TEST_F(AcyclicGraphTest, evaluate) {
  Eigen::ArrayXXd y = Evaluate(stack, x, constants);
  ASSERT_DOUBLE_EQ(y(0), 11.14);
  ASSERT_DOUBLE_EQ(y(1), 13.14);
  ASSERT_DOUBLE_EQ(y(2), 15.14);
}


TEST_F(AcyclicGraphTest, simplify) {
  // shorter stack
  CommandStack short_stack = SimplifyStack(stack);
  ASSERT_LE(short_stack.size(), stack.size());
  // equivalent evatuation
  Eigen::ArrayXXd y = Evaluate(stack, x, constants);
  Eigen::ArrayXXd simplified_y = SimplifyAndEvaluate(stack, x, constants);

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

  ASSERT_EQ(num_used_commands, 4);
}


}  // namespace

