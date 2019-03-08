/*!
 * \file agcpp_tests.cc
 *
 * \author Ethan Adams
 * \date
 *
 * This file contains the unit tests for the functions associated with the
 * AcyclicGraph and AcyclicGraphmanipulator class.
 */

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <set>
#include <string>
#include <sstream>

#include "gtest/gtest.h"

#include "test_fixtures.h"
#include "BingoCpp/acyclic_graph.h"
#include "BingoCpp/graph_manip.h"

using namespace bingo;

namespace {

class AGraphManipTest : public::testing::Test {
 public:
  AcyclicGraph test_indv;
  Eigen::ArrayX3i test_stack;
  Eigen::ArrayXXd test_x_vals;
  Eigen::ArrayXd temp_con;
  AcyclicGraphManipulator test_manip;

  void SetUp() {
    test_stack = testutils::stack_operators_0_to_5();
    test_x_vals = testutils::one_to_nine_3_by_3();
    temp_con = testutils::pi_ten_constants();

    int loads = 1;
    int stack_size = test_stack.rows();
    int nvars = 3;
    test_manip = AcyclicGraphManipulator(nvars, stack_size, loads);
    test_indv = AcyclicGraph();

    test_indv.stack = test_stack;
    test_manip.simplify_stack(test_indv);
    test_indv.set_constants(temp_con);
    test_manip.simplify_stack(test_indv);
  }

  void TearDown() {}
};

TEST_F(AGraphManipTest, add_node_type) {
  std::vector<int> truth{0, 1, 2, 3, 4, 5};
  
  test_manip.add_node_type(2);
  test_manip.add_node_type(3);
  test_manip.add_node_type(4);
  test_manip.add_node_type(5);

  for (int i = 0; i < truth.size(); ++i) {
    ASSERT_DOUBLE_EQ(truth[i], test_manip.node_type_vec[i]);
  }
}

TEST_F(AGraphManipTest, generate) {
  test_manip.add_node_type(2);
  test_manip.add_node_type(3);
  test_manip.add_node_type(4);
  test_manip.add_node_type(5);
  AcyclicGraph indv2 = test_manip.generate();
  ASSERT_DOUBLE_EQ(indv2.stack.rows(), 12);
}

TEST_F(AGraphManipTest, simplify_stack) {
    ASSERT_DOUBLE_EQ(test_indv.simple_stack.rows(), 8);
}

TEST_F(AGraphManipTest, dump) {
  std::pair<std::pair<Eigen::ArrayX3i, Eigen::VectorXd>, int> temp = test_manip.dump(
        test_indv);

  for (int i = 0; i < test_indv.stack.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test_indv.stack(i, 0), temp.first.first(i, 0));
    ASSERT_DOUBLE_EQ(test_indv.stack(i, 1), temp.first.first(i, 1));
    ASSERT_DOUBLE_EQ(test_indv.stack(i, 2), temp.first.first(i, 2));
  }

  for (int i = 0; i < test_indv.constants.size(); ++i) {
    ASSERT_DOUBLE_EQ(test_indv.constants(i), temp.first.second(i));
  }

  ASSERT_EQ(test_indv.genetic_age, temp.second);
}

TEST_F(AGraphManipTest, load) {
  std::pair<Eigen::ArrayX3i, Eigen::VectorXd> temp_pair(test_indv.stack,
      test_indv.constants);
  std::pair<std::pair<Eigen::ArrayX3i, Eigen::VectorXd>, int> truth(temp_pair,
      test_indv.genetic_age);
  AcyclicGraph temp = test_manip.load(truth);

  for (int i = 0; i < test_indv.stack.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test_indv.stack(i, 0), temp.stack(i, 0));
    ASSERT_DOUBLE_EQ(test_indv.stack(i, 1), temp.stack(i, 1));
    ASSERT_DOUBLE_EQ(test_indv.stack(i, 2), temp.stack(i, 2));
  }

  for (int i = 0; i < test_indv.constants.size(); ++i) {
    ASSERT_DOUBLE_EQ(test_indv.constants(i), temp.constants(i));
  }
}

TEST_F(AGraphManipTest, crossover) {
  test_manip.add_node_type(2);
  test_manip.add_node_type(3);
  test_manip.add_node_type(4);
  test_manip.add_node_type(5);
  AcyclicGraph indv2 = AcyclicGraph();
  Eigen::ArrayX3i stack3(12, 3);
  stack3 << 0, 0, 0,
            0, 1, 1,
            1, 0, 0,
            1, 1, 1,
            5, 3, 1,
            5, 3, 1,
            2, 4, 2,
            2, 3, 1,
            4, 5, 3,
            4, 4, 2,
            3, 6, 2,
            3, 8, 5;

  test_indv.set_constants(temp_con);
  indv2.stack = stack3;
  indv2.set_constants(temp_con);
  test_manip.simplify_stack(test_indv);
  test_manip.simplify_stack(indv2);
  std::vector<AcyclicGraph> children = test_manip.crossover(test_indv, indv2);
  AcyclicGraph c1 = children[0];
  AcyclicGraph c2 = children[1];
  bool all_match = true;

  for (int i = 0; i < 12; ++i) {
    if ((c1.stack(i, 0) != test_indv.stack(i, 0)) ||
        (c1.stack(i, 1) != test_indv.stack(i, 1)) ||
        (c1.stack(i, 2) != test_indv.stack(i, 2))) {
      all_match = false;
    }
  }

  EXPECT_FALSE(all_match);
}

TEST_F(AGraphManipTest, mutation) {
  test_manip.add_node_type(2);
  test_manip.add_node_type(3);
  test_manip.add_node_type(4);
  test_manip.add_node_type(5);
  AcyclicGraph indv2 = AcyclicGraph(test_indv);

  test_manip.mutation(indv2);
  bool all_match = true;

  for (int i = 0; i < test_indv.stack.rows(); ++i) {
    if ((test_indv.stack(i, 0) != indv2.stack(i, 0)) ||
        (test_indv.stack(i, 2) != indv2.stack(i, 2)) ||
        (test_indv.stack(i, 1) != indv2.stack(i, 1))) {
      all_match = false;
    }
  }

  all_match = false;
  EXPECT_FALSE(all_match);
}

TEST_F(AGraphManipTest, zero_distance) {
  AcyclicGraph indv2 = AcyclicGraph(test_indv);
  ASSERT_DOUBLE_EQ(test_manip.distance(test_indv, indv2), 0);
}

TEST_F(AGraphManipTest, rand_operator_params) {
  std::vector<int> term = test_manip.rand_operator_params(2, 0);
  std::vector<int> truth;
  truth.push_back(0);
  truth.push_back(0);

  for (int i = 0; i < 2; ++i) {
    ASSERT_DOUBLE_EQ(truth[i], term[i]);
  }

  std::vector<int> truth2;
  truth2.push_back(6);
  truth2.push_back(6);
  std::vector<int> op;
  op.push_back(6);
  op.push_back(6);
  op = test_manip.rand_operator_params(2, 6);

  for (int i = 0; i < 2; ++i) {
    ASSERT_NE(truth2[i], op[i]);
  }
}

TEST_F(AGraphManipTest, rand_operator_type) {
  test_manip.add_node_type(2);
  test_manip.add_node_type(3);
  test_manip.add_node_type(4);
  test_manip.add_node_type(5);
  test_manip.add_node_type(6);
  int op = test_manip.rand_operator_type();
  ASSERT_GT(op, 1);
  ASSERT_LT(op, 7);
}

TEST_F(AGraphManipTest, rand_operator) {
  test_manip.add_node_type(2);
  test_manip.add_node_type(3);
  test_manip.add_node_type(4);
  test_manip.add_node_type(5);
  test_manip.add_node_type(6);
  std::vector<int> term = test_manip.rand_operator(0);
  ASSERT_GE(term[0], 2);
  ASSERT_LE(term[0], 6);
  ASSERT_DOUBLE_EQ(term[1], 0);
  ASSERT_DOUBLE_EQ(term[2], 0);
  std::vector<int> op = test_manip.rand_operator(10);
  ASSERT_GE(term[0], 2);
  ASSERT_LE(term[0], 6);
  ASSERT_GE(term[1], 0);
  ASSERT_LE(term[1], 9);
  ASSERT_GE(term[2], 0);
  ASSERT_LE(term[2], 9);
}

TEST_F(AGraphManipTest, rand_terminal_param) {
  int termX = test_manip.rand_terminal_param(0);
  int termC = test_manip.rand_terminal_param(1);
  ASSERT_GE(termX, 0);
  ASSERT_LT(termX, 3);
  ASSERT_DOUBLE_EQ(termC, -1);
}

TEST_F(AGraphManipTest, mutate_terminal_param) {
  int termX = test_manip.rand_terminal_param(0);
  int termC = test_manip.rand_terminal_param(1);
  ASSERT_GE(termX, 0);
  ASSERT_LE(termX, 2);
  ASSERT_DOUBLE_EQ(termC, -1);
}

TEST_F(AGraphManipTest, rand_terminal) {
  std::vector<int> test = test_manip.rand_terminal();
  int node = test[0];
  ASSERT_GE(node, 0);
  ASSERT_LE(node, 1);

  if (node == 0) {
    ASSERT_GE(test[1], 0);
    ASSERT_LE(test[1], 2);
    ASSERT_GE(test[2], 0);
    ASSERT_LE(test[2], 2);

  } else {
    ASSERT_DOUBLE_EQ(test[1], -1);
    ASSERT_DOUBLE_EQ(test[2], -1);
  }
}
} // namespace