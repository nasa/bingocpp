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

  AcyclicGraphTest(): stack(12, 3), x(3, 3), stack2(2, 3), constants() {
  }

  void SetUp() {
    // y = x_0 * ( C_0 + C_1/x_1 ) - x_0
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
    // y = x_0 * x_0
    stack2 << 0, 0, 0,
           4, 0, 0;
    x << 1., 4., 7., 2., 5., 8., 3., 6., 9.;
    constants.push_back(3.14);
    constants.push_back(10.0);
  }

  void TearDown() {
    // code here will be called just after the test completes
  }
};

TEST(AcyclicGraphNodesTest, XLoad) {
  CommandStack stack(1, 3);
  Eigen::ArrayXXd x(3, 3);
  std::vector<double> constants;
  stack << 0, 1, 1;
  x << 7., 6., 9., 5., 11., 4., 3., 2., 1.;
  Eigen::ArrayXXd a_true(3, 1);
  a_true << 6., 11., 2.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
  }
}

TEST(AcyclicGraphNodesTest, CLoad) {
  CommandStack stack(1, 3);
  Eigen::ArrayXXd x(3, 3);
  std::vector<double> constants;
  stack << 1, 1, 1;
  constants.push_back(3.);
  constants.push_back(5.);
  Eigen::ArrayXXd a_true(3, 1);
  a_true << 5., 5., 5.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
  }
}

TEST(AcyclicGraphNodesTest, Addition) {
  CommandStack stack(3, 3);
  CommandStack stack2(3, 3);
  CommandStack stack3(2, 3);
  CommandStack stack4(2, 3);
  Eigen::ArrayXXd x(3, 3);
  std::vector<double> constants;
  stack << 0, 0, 0,
        1, 0, 0,
        2, 0, 1;
  stack2 << 0, 0, 0,
         1, 0, 0,
         2, 1, 0;
  stack3 << 0, 0, 0,
         2, 0, 0;
  stack4 << 1, 0, 0,
         2, 0, 0;
  x << 7., 5., 3., 6., 11., 4., 9., 8., 2.;
  constants.push_back(3.);
  Eigen::ArrayXXd a_true(3, 1);
  a_true << 10., 9., 12.;
  Eigen::ArrayXXd a_true_xx(3, 1);
  a_true_xx << 14., 12., 18.;
  Eigen::ArrayXXd a_true_cc(3, 1);
  a_true_cc << 6., 6., 6.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd test3 = Evaluate(stack3, x, constants);
  Eigen::ArrayXXd test4 = Evaluate(stack4, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 1., 0., 0., 1., 0., 0., 1., 0., 0.;
  Eigen::ArrayXXd d_true_xx(3, 3);
  d_true_xx << 2., 0., 0., 2., 0., 0., 2., 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xc =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_cx =
    EvaluateWithDerivative(stack2, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xx =
    EvaluateWithDerivative(stack3, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true(i));
    ASSERT_DOUBLE_EQ(test3(i), a_true_xx(i));
    ASSERT_DOUBLE_EQ(test4(i), a_true_cc(i));
    ASSERT_DOUBLE_EQ(d_xc.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_cx.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_xx.first(i), a_true_xx(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_xc.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_cx.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_xx.second(i), d_true_xx(i));
  }
}

TEST(AcyclicGraphNodesTest, Subtraction) {
  CommandStack stack(3, 3);
  CommandStack stack2(3, 3);
  CommandStack stack3(2, 3);
  CommandStack stack4(2, 3);
  Eigen::ArrayXXd x(3, 3);
  std::vector<double> constants;
  stack << 0, 0, 0,
        1, 0, 0,
        3, 0, 1;
  stack2 << 0, 0, 0,
         1, 0, 0,
         3, 1, 0;
  stack3 << 0, 2, 2,
         3, 0, 0;
  stack4 << 1, 0, 0,
         3, 0, 0;
  x << 7., 5., 3., 6., 11., 4., 9., 8., 2.;
  constants.push_back(3.);
  Eigen::ArrayXXd a_true(3, 1);
  a_true << 4., 3., 6.;
  Eigen::ArrayXXd a_true2(3, 1);
  a_true2 << -4., -3., -6.;
  Eigen::ArrayXXd a_true_xx(3, 1);
  a_true_xx << 0., 0., 0.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd test3 = Evaluate(stack3, x, constants);
  Eigen::ArrayXXd test4 = Evaluate(stack4, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 1., 0., 0., 1., 0., 0., 1., 0., 0.;
  Eigen::ArrayXXd d_true2(3, 3);
  d_true2 << -1., 0., 0., -1., 0., 0., -1., 0., 0.;
  Eigen::ArrayXXd d_true_xx(3, 3);
  d_true_xx << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xc =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_cx =
    EvaluateWithDerivative(stack2, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xx =
    EvaluateWithDerivative(stack3, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true2(i));
    ASSERT_DOUBLE_EQ(test3(i), a_true_xx(i));
    ASSERT_DOUBLE_EQ(test4(i), a_true_xx(i));
    ASSERT_DOUBLE_EQ(d_xc.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_cx.first(i), a_true2(i));
    ASSERT_DOUBLE_EQ(d_xx.first(i), a_true_xx(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_xc.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_cx.second(i), d_true2(i));
    ASSERT_DOUBLE_EQ(d_xx.second(i), d_true_xx(i));
  }
}

TEST(AcyclicGraphNodesTest, Multiplication) {
  CommandStack stack(3, 3);
  CommandStack stack2(3, 3);
  CommandStack stack3(2, 3);
  CommandStack stack4(2, 3);
  Eigen::ArrayXXd x(3, 3);
  std::vector<double> constants;
  stack << 0, 0, 0,
        1, 0, 0,
        4, 0, 1;
  stack2 << 0, 0, 0,
         1, 0, 0,
         4, 1, 0;
  stack3 << 0, 0, 0,
         4, 0, 0;
  stack4 << 1, 0, 0,
         4, 0, 0;
  x << 7., 5., 3., 6., 11., 4., 9., 8., 2.;
  constants.push_back(3.);
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (7. * 3.), (6. * 3.), (9. * 3.);
  Eigen::ArrayXXd a_true_xx(3, 1);
  a_true_xx << (7. * 7.), (6. * 6.), (9.* 9.);
  Eigen::ArrayXXd a_true_cc(3, 1);
  a_true_cc << 9., 9., 9.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd test3 = Evaluate(stack3, x, constants);
  Eigen::ArrayXXd test4 = Evaluate(stack4, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 3., 0., 0., 3., 0., 0., 3., 0., 0.;
  Eigen::ArrayXXd d_true_xx(3, 3);
  d_true_xx << 14., 0., 0., 12., 0., 0., 18., 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xc =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_cx =
    EvaluateWithDerivative(stack2, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xx =
    EvaluateWithDerivative(stack3, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true(i));
    ASSERT_DOUBLE_EQ(test3(i), a_true_xx(i));
    ASSERT_DOUBLE_EQ(test4(i), a_true_cc(i));
    ASSERT_DOUBLE_EQ(d_xc.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_cx.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_xx.first(i), a_true_xx(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_xc.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_cx.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_xx.second(i), d_true_xx(i));
  }
}

TEST(AcyclicGraphNodesTest, Division) {
  CommandStack stack(3, 3);
  CommandStack stack2(3, 3);
  CommandStack stack3(2, 3);
  CommandStack stack4(2, 3);
  Eigen::ArrayXXd x(3, 3);
  std::vector<double> constants;
  stack << 0, 0, 0,
        1, 0, 0,
        5, 0, 1;
  stack2 << 0, 0, 0,
         1, 0, 0,
         5, 1, 0;
  stack3 << 0, 2, 2,
         5, 0, 0;
  stack4 << 1, 0, 0,
         5, 0, 0;
  x << 7., 5., 3., 6., 11., 12., 9., 8., 6.;
  constants.push_back(3.);
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (7. / 3.), (6. / 3.), (9. / 3.);
  Eigen::ArrayXXd a_true2(3, 1);
  a_true2 << (3. / 7.), (3. / 6.), (3. / 9.);
  Eigen::ArrayXXd a_true_xx(3, 1);
  a_true_xx << 1., 1., 1.;
  Eigen::ArrayXXd a_true_cc(3, 1);
  a_true_cc << 1., 1., 1.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd test3 = Evaluate(stack3, x, constants);
  Eigen::ArrayXXd test4 = Evaluate(stack4, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << (1. / 3.), 0., 0., (1. / 3.), 0., 0., (1. / 3.), 0., 0.;
  Eigen::ArrayXXd d_true2(3, 3);
  d_true2 << (-3. / (pow(7., 2.))), 0., 0.,
          (-3. / (pow(6., 2.))), 0., 0.,
          (-3. / (pow(9., 2.))), 0., 0.;
  Eigen::ArrayXXd d_true_xx(3, 3);
  d_true_xx << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xc =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_cx =
    EvaluateWithDerivative(stack2, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xx =
    EvaluateWithDerivative(stack3, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true2(i));
    ASSERT_DOUBLE_EQ(test3(i), a_true_xx(i));
    ASSERT_DOUBLE_EQ(test4(i), a_true_cc(i));
    ASSERT_DOUBLE_EQ(d_xc.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_cx.first(i), a_true2(i));
    ASSERT_DOUBLE_EQ(d_xx.first(i), a_true_xx(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_xc.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_cx.second(i), d_true2(i));
    ASSERT_DOUBLE_EQ(d_xx.second(i), d_true_xx(i));
  }
}

TEST(AcyclicGraphNodesTest, Sin) {
  CommandStack stack(2, 3);
  CommandStack stack2(2, 3);
  Eigen::ArrayXXd x(3, 3);
  std::vector<double> constants;
  stack << 1, 0, 0,
        6, 0, 0;
  stack2 << 0, 0, 0,
         6, 0, 0;
  x << 7., 5., 3., 6., 11., 12., 9., 8., 6.;
  constants.push_back(3.);
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (sin(3.)), (sin(3.)), (sin(3.));
  Eigen::ArrayXXd a_true_x(3, 1);
  a_true_x << (sin(7.)), (sin(6.)), (sin(9.));
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  Eigen::ArrayXXd d_true_x(3, 3);
  d_true_x << (cos(7.)), 0., 0.,
           (cos(6.)), 0., 0.,
           (cos(9.)), 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_c =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_x =
    EvaluateWithDerivative(stack2, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true_x(i));
    ASSERT_DOUBLE_EQ(d_c.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_x.first(i), a_true_x(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_c.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_x.second(i), d_true_x(i));
  }
}

TEST(AcyclicGraphNodesTest, Cos) {
  CommandStack stack(2, 3);
  CommandStack stack2(2, 3);
  Eigen::ArrayXXd x(3, 3);
  std::vector<double> constants;
  stack << 1, 0, 0,
        7, 0, 0;
  stack2 << 0, 0, 0,
         7, 0, 0;
  x << 7., 5., 3., 6., 11., 12., 9., 8., 6.;
  constants.push_back(3.);
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (cos(3.)), (cos(3.)), (cos(3.));
  Eigen::ArrayXXd a_true_x(3, 1);
  a_true_x << (cos(7.)), (cos(6.)), (cos(9.));
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  Eigen::ArrayXXd d_true_x(3, 3);
  d_true_x << (-sin(7.)), 0., 0.,
           (-sin(6.)), 0., 0.,
           (-sin(9.)), 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_c =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_x =
    EvaluateWithDerivative(stack2, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true_x(i));
    ASSERT_DOUBLE_EQ(d_c.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_x.first(i), a_true_x(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_c.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_x.second(i), d_true_x(i));
  }
}

TEST(AcyclicGraphNodesTest, Exp) {
  CommandStack stack(2, 3);
  CommandStack stack2(2, 3);
  Eigen::ArrayXXd x(3, 3);
  std::vector<double> constants;
  stack << 1, 0, 0,
        8, 0, 0;
  stack2 << 0, 0, 0,
         8, 0, 0;
  x << 7., 5., 3., 6., 11., 12., 9., 8., 6.;
  constants.push_back(3.);
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (exp(3.)), (exp(3.)), (exp(3.));
  Eigen::ArrayXXd a_true_x(3, 1);
  a_true_x << (exp(7.)), (exp(6.)), (exp(9.));
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  Eigen::ArrayXXd d_true_x(3, 3);
  d_true_x << (exp(7.)), 0., 0.,
           (exp(6.)), 0., 0.,
           (exp(9.)), 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_c =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_x =
    EvaluateWithDerivative(stack2, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true_x(i));
    ASSERT_DOUBLE_EQ(d_c.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_x.first(i), a_true_x(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_c.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_x.second(i), d_true_x(i));
  }
}

TEST(AcyclicGraphNodesTest, Log) {
  CommandStack stack(2, 3);
  CommandStack stack2(2, 3);
  Eigen::ArrayXXd x(3, 3);
  std::vector<double> constants;
  stack << 1, 0, 0,
        9, 0, 0;
  stack2 << 0, 0, 0,
         9, 0, 0;
  x << 7., 5., 3., 6., 11., 12., 9., 8., 6.;
  constants.push_back(3.);
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (log(abs(3.))), (log(abs(3.))), (log(abs(3.)));
  Eigen::ArrayXXd a_true_x(3, 1);
  a_true_x << (log(abs(7.))), (log(abs(6.))), (log(abs(9.)));
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  Eigen::ArrayXXd d_true_x(3, 3);
  d_true_x << (1. / abs(7.)), 0., 0.,
           (1. / abs(6.)), 0., 0.,
           (1. / abs(9.)), 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_c =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_x =
    EvaluateWithDerivative(stack2, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true_x(i));
    ASSERT_DOUBLE_EQ(d_c.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_x.first(i), a_true_x(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_c.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_x.second(i), d_true_x(i));
  }
}

TEST(AcyclicGraphNodesTest, Power) {
  CommandStack stack(3, 3);
  CommandStack stack2(3, 3);
  CommandStack stack3(2, 3);
  CommandStack stack4(2, 3);
  Eigen::ArrayXXd x(3, 3);
  std::vector<double> constants;
  stack << 0, 0, 0,
        1, 0, 0,
        10, 0, 1;
  stack2 << 0, 0, 0,
         1, 0, 0,
         10, 1, 0;
  stack3 << 0, 0, 0,
         10, 0, 0;
  stack4 << 1, 0, 0,
         10, 0, 0;
  x << 7., 5., 3., 6., 11., 4., 9., 8., 6.;
  constants.push_back(3.);
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (pow(7., 3.)), (pow(6., 3.)), (pow(9., 3.));
  Eigen::ArrayXXd a_true2(3, 1);
  a_true2 << (pow(3., 7.)), (pow(3., 6.)), (pow(3., 9.));
  Eigen::ArrayXXd a_true_xx(3, 1);
  a_true_xx << (pow(7., 7.)), (pow(6., 6.)), (pow(9., 9.));
  Eigen::ArrayXXd a_true_cc(3, 1);
  a_true_cc << (pow(3., 3.)), (pow(3., 3.)), (pow(3., 3.));
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd test3 = Evaluate(stack3, x, constants);
  Eigen::ArrayXXd test4 = Evaluate(stack4, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << (3. * pow(7., 2.)), 0., 0.,
         (3. * pow(6., 2.)), 0., 0.,
         (3. * pow(9., 2.)), 0., 0.;
  Eigen::ArrayXXd d_true2(3, 3);
  d_true2 << (pow(3., 7.) * log(3.)), 0., 0.,
          (pow(3., 6.) * log(3.)), 0., 0.,
          (pow(3., 9.) * log(3.)), 0., 0.;
  Eigen::ArrayXXd d_true_xx(3, 3);
  d_true_xx << (pow(7., 7.) * (1 + log(7.))), 0., 0.,
            (pow(6., 6.) * (1 + log(6.))), 0., 0.,
            (pow(9., 9.) * (1 + log(9.))), 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xc =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_cx =
    EvaluateWithDerivative(stack2, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_xx =
    EvaluateWithDerivative(stack3, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true2(i));
    ASSERT_DOUBLE_EQ(test3(i), a_true_xx(i));
    ASSERT_DOUBLE_EQ(test4(i), a_true_cc(i));
    ASSERT_DOUBLE_EQ(d_xc.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_cx.first(i), a_true2(i));
    ASSERT_DOUBLE_EQ(d_xx.first(i), a_true_xx(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_xc.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_cx.second(i), d_true2(i));
    ASSERT_DOUBLE_EQ(d_xx.second(i), d_true_xx(i));
  }
}

TEST(AcyclicGraphNodesTest, Absolute) {
  CommandStack stack(2, 3);
  CommandStack stack2(2, 3);
  Eigen::ArrayXXd x(3, 3);
  std::vector<double> constants;
  stack << 1, 0, 0,
        11, 0, 0;
  stack2 << 0, 0, 0,
         11, 0, 0;
  x << -7., 5., 3., 6., 11., 4., -9., 8., 6.;
  constants.push_back(-3.);
  Eigen::ArrayXXd a_true(3, 1);
  a_true << 3., 3., 3.;
  Eigen::ArrayXXd a_true_x(3, 1);
  a_true_x << 7., 6., 9.;
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  Eigen::ArrayXXd d_true_x(3, 3);
  d_true_x << -1., 0., 0., 1., 0., 0., -1., 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_c =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_x =
    EvaluateWithDerivative(stack2, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true_x(i));
    ASSERT_DOUBLE_EQ(d_c.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_x.first(i), a_true_x(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_c.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_x.second(i), d_true_x(i));
  }
}

TEST(AcyclicGraphNodesTest, Sqrt) {
  CommandStack stack(2, 3);
  CommandStack stack2(2, 3);
  Eigen::ArrayXXd x(3, 3);
  std::vector<double> constants;
  stack << 1, 0, 0,
        12, 0, 0;
  stack2 << 0, 0, 0,
         12, 0, 0;
  x << -7., 5., 3., 6., 11., 4., -9., 8., 6.;
  constants.push_back(3.);
  Eigen::ArrayXXd a_true(3, 1);
  a_true << (sqrt(abs(3.))), (sqrt(abs(3.))), (sqrt(abs(3.)));
  Eigen::ArrayXXd a_true_x(3, 1);
  a_true_x << (sqrt(abs(-7.))), (sqrt(abs(6.))), (sqrt(abs(-9.)));
  Eigen::ArrayXXd test = Evaluate(stack, x, constants);
  Eigen::ArrayXXd test2 = Evaluate(stack2, x, constants);
  Eigen::ArrayXXd d_true(3, 3);
  d_true << 0., 0., 0., 0., 0., 0., 0., 0., 0.;
  Eigen::ArrayXXd d_true_x(3, 3);
  d_true_x << 1. / (2.*sqrt(7.)), 0., 0.,
           1. / (2.*sqrt(6.)), 0., 0.,
           1. / (2.*sqrt(9.)), 0., 0.;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_c =
    EvaluateWithDerivative(stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> d_x =
    EvaluateWithDerivative(stack2, x, constants);

  for (size_t i = 0; i < x.rows(); ++i) {
    ASSERT_DOUBLE_EQ(test(i), a_true(i));
    ASSERT_DOUBLE_EQ(test2(i), a_true_x(i));
    ASSERT_DOUBLE_EQ(d_c.first(i), a_true(i));
    ASSERT_DOUBLE_EQ(d_x.first(i), a_true_x(i));
  }

  for (size_t i = 0; i < x.size(); ++i) {
    ASSERT_DOUBLE_EQ(d_c.second(i), d_true(i));
    ASSERT_DOUBLE_EQ(d_x.second(i), d_true_x(i));
  }
}

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
  ASSERT_LE(short_stack.rows(), stack.rows());
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



