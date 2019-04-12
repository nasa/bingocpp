/*!
 * \file agcpp_tests.cc
 *
 * \author Ethan Adams
 * \date
 *
 * This file contains the unit tests for the functions associated with the
 * AcyclicGraph and AcyclicGraphmanipulator class.
 */

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

class AcylicGraphTest : public::testing::Test {
 public:
  AcyclicGraph test_indv;
  Eigen::ArrayX3i test_stack;
  Eigen::ArrayXXd test_x_vals;
  Eigen::ArrayXXd temp_con;
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

TEST_F(AcylicGraphTest, utilized_commands) {
  std::set<int> x = test_indv.utilized_commands();
  std::set<int> x_true = {0, 1, 2, 3, 4, 6, 8, 11};
  std::set<int>::iterator it2 = x_true.begin();

  for (std::set<int>::iterator it = x.begin(); it != x.end(); ++it, ++it2) {
    ASSERT_DOUBLE_EQ(*it, *it2);
  }
}

TEST_F(AcylicGraphTest, copy) {
  AcyclicGraph indv2 = AcyclicGraph(test_indv);

  for (size_t i = 0; i < indv2.stack.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test_indv.stack(i, 0), indv2.stack(i, 0));
    ASSERT_DOUBLE_EQ(test_indv.stack(i, 1), indv2.stack(i, 1));
    ASSERT_DOUBLE_EQ(test_indv.stack(i, 2), indv2.stack(i, 2));
  }
}

TEST_F(AcylicGraphTest, needs_optimization) {
  EXPECT_FALSE(test_indv.needs_optimization());
}

TEST_F(AcylicGraphTest, set_constants) {
  Eigen::VectorXd con(2);
  con << 12.0, 5.0;
  test_indv.set_constants(con);

  for (int i = 0; i < 2; ++i) {
    ASSERT_DOUBLE_EQ(test_indv.constants[i], con[i]);
  }
}

TEST_F(AcylicGraphTest, count_constants) {
  ASSERT_EQ(test_indv.count_constants(), 2);
}

TEST_F(AcylicGraphTest, evaluate) {
  std::vector<double> truth{4.64, 8.28, 11.42};
  Eigen::ArrayXXd eig = test_indv.evaluate(test_x_vals);
  for (int i = 0; i < 3; ++i) {
    ASSERT_NEAR(eig(i), truth[i], .001);
  }
}

TEST_F(AcylicGraphTest, evaluate_deriv) {
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> p = 
    test_indv.evaluate_deriv(test_x_vals);
  ASSERT_NEAR(p.first(0), 4.64, .001);
  ASSERT_NEAR(p.first(1), 8.28, .001);
  ASSERT_NEAR(p.first(2), 11.42, .001);
  ASSERT_NEAR(p.second(0), 4.64, .001);
  ASSERT_NEAR(p.second(1), 4.14, .001);
  ASSERT_NEAR(p.second(2), 3.806, .001);
}

TEST_F(AcylicGraphTest, latexstring) {
  std::string str_true = "(\\frac{10}{X_1} + 3.14)(X_0) - (X_0)";
  EXPECT_EQ(str_true, test_indv.latexstring());
}

TEST_F(AcylicGraphTest, complexity) {
  EXPECT_EQ(8, test_indv.complexity());
}

TEST_F(AcylicGraphTest, print_stack) {
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
  EXPECT_EQ(out.str(), test_indv.print_stack());
}
} // namespace

// TEST_F(AcylicGraphTest, input_constants) {
//     Eigen::ArrayX3d stack(12,3);
//     stack << 0, 0, 0,
//               0, 1, 1,
//               1, -1, -1,
//               1, -1, -1,
//               5, 3, 1,
//               5, 3, 1,
//               2, 4, 2,
//               2, 4, 2,
//               4, 6, 0,
//               4, 5, 6,
//               3, 7, 6,
//               3, 8, 0;
//     indv.stack = stack;
//     AcyclicGraphManipulator manip = AcyclicGraphManipulator(3, 12, 1);
//     manip.simplify_stack(indv);
//     indv.input_constants();
//     bool fail = false;
//     for (int i = 0; i < indv.simple_stack.rows(); ++i) {
//         if (indv.simple_stack(i, 0) == 1 && indv.simple_stack(i, 1) == -1)
//             fail = true;
//     }
//     ASSERT_EQ(fail, false);
// }
