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
#include "BingoCpp/graph_manip.h"

class AgcppTest : public::testing::Test {
 public:
  AcyclicGraph indv;
  Eigen::ArrayX3i stack2;
  Eigen::ArrayXXd x;
  AcyclicGraphManipulator manip;

  AgcppTest(): stack2(12, 3), x(3, 3) {
  }

  void SetUp() {
    manip = AcyclicGraphManipulator(3, 12, 1);
    indv = AcyclicGraph();
    stack2 << 0, 0, 0,
           0, 1, 1,
           1, 0, 0,
           1, 1, 1,
           5, 3, 1,
           5, 3, 1,
           2, 4, 2,
           2, 4, 2,
           4, 6, 0,
           4, 5, 6,
           3, 7, 6,
           3, 8, 0;
    x << 1., 4., 7., 2., 5., 8., 3., 6., 9.;
    indv.stack = stack2;
    Eigen::VectorXd temp_con(2);
    temp_con << 3.14, 10.0;
    indv.set_constants(temp_con);
    manip.simplify_stack(indv);
  }

  void TearDown() {
    // code here will be called just after the test completes
  }
};

class AgcppManipTest : public::testing::Test {
 public:

  AgcppManipTest() {}
  virtual ~AgcppManipTest() {}

  AcyclicGraph indv;
  AcyclicGraphManipulator manip;

  void SetUp() {
    manip = AcyclicGraphManipulator(3, 12, 1);
    Eigen::ArrayX3i stack(12, 3);
    Eigen::ArrayXXd x(3, 3);
    indv = AcyclicGraph();
    stack << 0, 0, 0,
          0, 1, 1,
          1, 0, 0,
          1, 1, 1,
          5, 3, 1,
          5, 3, 1,
          2, 4, 2,
          2, 4, 2,
          4, 6, 0,
          4, 5, 6,
          3, 7, 6,
          3, 8, 0;
    x << 1., 4., 7., 2., 5., 8., 3., 6., 9.;
    indv.stack = stack;
    Eigen::VectorXd temp_con(2);
    temp_con << 3.14, 10.0;
    indv.set_constants(temp_con);
    manip.simplify_stack(indv);
  }

  void TearDown() {
    // code here will be called just after the test completes
  }
};

TEST_F(AgcppTest, utilized_commands) {
  std::set<int> x = indv.utilized_commands();
  std::set<int> x_true = {0, 1, 2, 3, 4, 6, 8, 11};
  std::set<int>::iterator it2 = x_true.begin();

  for (std::set<int>::iterator it = x.begin(); it != x.end(); ++it, ++it2) {
    ASSERT_DOUBLE_EQ(*it, *it2);
  }
}

TEST_F(AgcppTest, copy) {
  AcyclicGraph indv2 = AcyclicGraph(indv);

  for (size_t i = 0; i < indv.stack.rows(); ++i) {
    ASSERT_DOUBLE_EQ(indv.stack(i, 0), indv2.stack(i, 0));
    ASSERT_DOUBLE_EQ(indv.stack(i, 1), indv2.stack(i, 1));
    ASSERT_DOUBLE_EQ(indv.stack(i, 2), indv2.stack(i, 2));
  }
}

TEST_F(AgcppTest, needs_optimization) {
  EXPECT_FALSE(indv.needs_optimization());
}

TEST_F(AgcppTest, set_constants) {
  Eigen::VectorXd con(2);
  con << 12.0, 5.0;
  indv.set_constants(con);

  for (int i = 0; i < 2; ++i) {
    ASSERT_DOUBLE_EQ(indv.constants[i], con[i]);
  }
}

TEST_F(AgcppTest, count_constants) {
  ASSERT_EQ(indv.count_constants(), 2);
}

TEST_F(AgcppTest, evaluate) {
  std::vector<double> truth;
  truth.push_back(4.64);
  truth.push_back(8.28);
  truth.push_back(11.42);
  Eigen::ArrayXXd eig = indv.evaluate(x);

  for (int i = 0; i < 3; ++i) {
    ASSERT_NEAR(eig(i), truth[i], .001);
  }
}

TEST_F(AgcppTest, evaluate_deriv) {
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> p = indv.evaluate_deriv(x);
  ASSERT_NEAR(p.first(0), 4.64, .001);
  ASSERT_NEAR(p.first(1), 8.28, .001);
  ASSERT_NEAR(p.first(2), 11.42, .001);
  ASSERT_NEAR(p.second(0), 4.64, .001);
  ASSERT_NEAR(p.second(1), 4.14, .001);
  ASSERT_NEAR(p.second(2), 3.806, .001);
}

TEST_F(AgcppTest, latexstring) {
  std::string str_true = "(\\frac{10}{X_1} + 3.14)(X_0) - (X_0)";
  EXPECT_EQ(str_true, indv.latexstring());
}

TEST_F(AgcppTest, complexity) {
  EXPECT_EQ(8, indv.complexity());
}

TEST_F(AgcppTest, print_stack) {
  std::ostringstream out;
  out << "---full stack---\n";
  out << 0 << "   <= X0\n";
  out << 1 << "   <= X1\n";
  out << 2 << "   <= 3.14\n";
  out << 3 << "   <= 10\n";
  out << 4 << "   <= (3) / (1)\n";
  out << 5 << "   <= (3) / (1)\n";
  out << 6 << "   <= (4) + (2)\n";
  out << 7 << "   <= (4) + (2)\n";
  out << 8 << "   <= (6) * (0)\n";
  out << 9 << "   <= (5) * (6)\n";
  out << 10 << "  <= (7) - (6)\n";
  out << 11 << "  <= (8) - (0)\n";
  out << "---small stack---\n";
  out << 0 << "   <= X0\n";
  out << 1 << "   <= X1\n";
  out << 2 << "   <= 3.14\n";
  out << 3 << "   <= 10\n";
  out << 4 << "   <= (3) / (1)\n";
  out << 5 << "   <= (4) + (2)\n";
  out << 6 << "   <= (5) * (0)\n";
  out << 7 << "   <= (6) - (0)\n";
  EXPECT_EQ(out.str(), indv.print_stack());
}

TEST_F(AgcppManipTest, add_node_type) {
  SetUp();
  std::vector<int> truth;
  truth.push_back(0);
  truth.push_back(1);
  truth.push_back(2);
  truth.push_back(3);
  truth.push_back(4);
  truth.push_back(5);
  manip.add_node_type(2);
  manip.add_node_type(3);
  manip.add_node_type(4);
  manip.add_node_type(5);

  for (int i = 0; i < truth.size(); ++i) {
    ASSERT_DOUBLE_EQ(truth[i], manip.node_type_vec[i]);
  }
}

TEST_F(AgcppManipTest, generate) {
  SetUp();
  manip.add_node_type(2);
  manip.add_node_type(3);
  manip.add_node_type(4);
  manip.add_node_type(5);
  AcyclicGraph indv2 = manip.generate();
  ASSERT_DOUBLE_EQ(indv2.stack.rows(), 12);
}

TEST_F(AgcppManipTest, dump) {
  SetUp();
  std::pair<std::pair<Eigen::ArrayX3i, Eigen::VectorXd>, int> temp = manip.dump(
        indv);

  for (int i = 0; i < indv.stack.rows(); ++i) {
    ASSERT_DOUBLE_EQ(indv.stack(i, 0), temp.first.first(i, 0));
    ASSERT_DOUBLE_EQ(indv.stack(i, 1), temp.first.first(i, 1));
    ASSERT_DOUBLE_EQ(indv.stack(i, 2), temp.first.first(i, 2));
  }

  for (int i = 0; i < indv.constants.size(); ++i) {
    ASSERT_DOUBLE_EQ(indv.constants(i), temp.first.second(i));
  }

  ASSERT_EQ(indv.genetic_age, temp.second);
}

TEST_F(AgcppManipTest, load) {
  SetUp();
  std::pair<Eigen::ArrayX3i, Eigen::VectorXd> temp_pair(indv.stack,
      indv.constants);
  std::pair<std::pair<Eigen::ArrayX3i, Eigen::VectorXd>, int> truth(temp_pair,
      indv.genetic_age);
  AcyclicGraph temp = manip.load(truth);

  for (int i = 0; i < indv.stack.rows(); ++i) {
    ASSERT_DOUBLE_EQ(indv.stack(i, 0), temp.stack(i, 0));
    ASSERT_DOUBLE_EQ(indv.stack(i, 1), temp.stack(i, 1));
    ASSERT_DOUBLE_EQ(indv.stack(i, 2), temp.stack(i, 2));
  }

  for (int i = 0; i < indv.constants.size(); ++i) {
    ASSERT_DOUBLE_EQ(indv.constants(i), temp.constants(i));
  }
}

TEST_F(AgcppManipTest, crossover) {
  SetUp();
  manip.add_node_type(2);
  manip.add_node_type(3);
  manip.add_node_type(4);
  manip.add_node_type(5);
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
  Eigen::VectorXd temp_con(2);
  temp_con << 3.14, 10.0;
  indv.set_constants(temp_con);
  indv2.stack = stack3;
  indv2.set_constants(temp_con);
  manip.simplify_stack(indv);
  manip.simplify_stack(indv2);
  std::vector<AcyclicGraph> children = manip.crossover(indv, indv2);
  AcyclicGraph c1 = children[0];
  AcyclicGraph c2 = children[1];
  bool all_match = true;

  for (int i = 0; i < 12; ++i) {
    if ((c1.stack(i, 0) != indv.stack(i, 0)) ||
        (c1.stack(i, 1) != indv.stack(i, 1)) ||
        (c1.stack(i, 2) != indv.stack(i, 2))) {
      all_match = false;
    }
  }

  EXPECT_FALSE(all_match);
}

TEST_F(AgcppManipTest, mutation) {
  SetUp();
  manip.add_node_type(2);
  manip.add_node_type(3);
  manip.add_node_type(4);
  manip.add_node_type(5);
  AcyclicGraph indv2 = AcyclicGraph(indv);
  // for (int i = 0; i < 10; ++i)
  manip.mutation(indv2);
  bool all_match = true;

  for (int i = 0; i < indv.stack.rows(); ++i) {
    if ((indv.stack(i, 0) != indv2.stack(i, 0)) ||
        (indv.stack(i, 2) != indv2.stack(i, 2)) ||
        (indv.stack(i, 1) != indv2.stack(i, 1))) {
      all_match = false;
    }
  }

  all_match = false;
  EXPECT_FALSE(all_match);
}

TEST_F(AgcppManipTest, distance) {
  SetUp();
  AcyclicGraph indv2 = AcyclicGraph(indv);
  ASSERT_DOUBLE_EQ(manip.distance(indv, indv2), 0);
}

TEST_F(AgcppManipTest, rand_operator_params) {
  SetUp();
  std::vector<int> term = manip.rand_operator_params(2, 0);
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
  op = manip.rand_operator_params(2, 6);

  for (int i = 0; i < 2; ++i) {
    ASSERT_NE(truth2[i], op[i]);
  }
}

TEST_F(AgcppManipTest, rand_operator_type) {
  SetUp();
  manip.add_node_type(2);
  manip.add_node_type(3);
  manip.add_node_type(4);
  manip.add_node_type(5);
  manip.add_node_type(6);
  int op = manip.rand_operator_type();
  ASSERT_GT(op, 1);
  ASSERT_LT(op, 7);
}

TEST_F(AgcppManipTest, rand_operator) {
  SetUp();
  manip.add_node_type(2);
  manip.add_node_type(3);
  manip.add_node_type(4);
  manip.add_node_type(5);
  manip.add_node_type(6);
  std::vector<int> term = manip.rand_operator(0);
  ASSERT_GE(term[0], 2);
  ASSERT_LE(term[0], 6);
  ASSERT_DOUBLE_EQ(term[1], 0);
  ASSERT_DOUBLE_EQ(term[2], 0);
  std::vector<int> op = manip.rand_operator(10);
  ASSERT_GE(term[0], 2);
  ASSERT_LE(term[0], 6);
  ASSERT_GE(term[1], 0);
  ASSERT_LE(term[1], 9);
  ASSERT_GE(term[2], 0);
  ASSERT_LE(term[2], 9);
}

TEST_F(AgcppManipTest, rand_terminal_param) {
  SetUp();
  int termX = manip.rand_terminal_param(0);
  int termC = manip.rand_terminal_param(1);
  ASSERT_GE(termX, 0);
  ASSERT_LT(termX, 3);
  ASSERT_DOUBLE_EQ(termC, -1);
}

TEST_F(AgcppManipTest, mutate_terminal_param) {
  SetUp();
  int termX = manip.rand_terminal_param(0);
  int termC = manip.rand_terminal_param(1);
  ASSERT_GE(termX, 0);
  ASSERT_LE(termX, 2);
  ASSERT_DOUBLE_EQ(termC, -1);
}

TEST_F(AgcppManipTest, rand_terminal) {
  SetUp();
  std::vector<int> test = manip.rand_terminal();
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