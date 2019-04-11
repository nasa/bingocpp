#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "BingoCpp/backend.h"
#include "testing_utils.h"
#include "test_fixtures.h"

using namespace bingo;
using namespace backend;

namespace {
const int N_OPS = 13;

class AGraphBackend : public ::testing::TestWithParam<int> {
 public:
  
  const double AGRAPH_VAL_START =-1;
  const double AGRAPH_VAL_END = 0;
  const int N_AGRAPH_VAL = 11;

  testutils::AGraphValues sample_agraph_1_values;
  std::vector<Eigen::ArrayXXd> operator_evals_x0;
  std::vector<Eigen::ArrayXXd> operator_x_derivs;
  std::vector<Eigen::ArrayXXd> operator_c_derivs;

  Eigen::ArrayX3i simple_stack;
  Eigen::ArrayX3i simple_stack2;
  Eigen::ArrayXXd x;
  Eigen::ArrayXd constants;

  virtual void SetUp() {
    sample_agraph_1_values = testutils::init_agraph_vals(AGRAPH_VAL_START,
                                                         AGRAPH_VAL_END,
                                                         N_AGRAPH_VAL);
    operator_evals_x0 = testutils::init_op_evals_x0(sample_agraph_1_values);
    operator_x_derivs = testutils::init_op_x_derivs(sample_agraph_1_values);
    operator_c_derivs = testutils::init_op_c_derivs(sample_agraph_1_values);

    simple_stack = testutils::stack_operators_0_to_5(); 
    simple_stack2 = testutils::stack_unary_operator(4);
    x = testutils::one_to_nine_3_by_3();
    constants = testutils::pi_ten_constants();
  }
  virtual void TearDown() {}
};

TEST_P(AGraphBackend, simplify_and_evaluate) {
  int operator_i = GetParam();
  Eigen::ArrayXXd expected_outcome = operator_evals_x0[operator_i];

  Eigen::ArrayX3i stack(3, 3);
  stack << 0, 0, 0,
           0, 1, 0,
           operator_i, 0, 0;
  Eigen::ArrayXXd f_of_x = simplifyAndEvaluate(stack,
                                               sample_agraph_1_values.x_vals,
                                               sample_agraph_1_values.constants);
  ASSERT_TRUE(testutils::almost_equal(expected_outcome, f_of_x));
}

TEST_P(AGraphBackend, simplify_and_evaluate_x_deriv) {
  int operator_i = GetParam();
  Eigen::ArrayXXd expected_derivative = 
    Eigen::ArrayXXd::Zero(sample_agraph_1_values.x_vals.rows(), 2);
  expected_derivative.col(0) = operator_x_derivs[operator_i];

  Eigen::ArrayX3i stack(4, 3);
  stack << 0, 0, 0,
           0, 0, 0,
           0, 1, 1,
           operator_i, 0, 1;

  Eigen::ArrayXXd x_0 = sample_agraph_1_values.x_vals;
  Eigen::ArrayXXd constants = sample_agraph_1_values.constants;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> res_and_gradient = 
    simplifyAndEvaluateWithDerivative(stack,
                                      x_0,
                                      constants,
                                      true);
  Eigen::ArrayXXd df_dx = res_and_gradient.second;
  ASSERT_TRUE(testutils::almost_equal(expected_derivative, df_dx));
}

TEST_P(AGraphBackend, simplify_and_evaluate_c_deriv) {
  int operator_i = GetParam();
  int num_x_points = sample_agraph_1_values.x_vals.rows();
  int num_consts = sample_agraph_1_values.constants.size();
  int last_col = num_consts - 1;
  Eigen::ArrayXXd expected_derivative = 
    Eigen::MatrixXd::Zero(num_x_points, num_consts).array();
  expected_derivative.col(last_col) = operator_c_derivs[operator_i];
  
  Eigen::ArrayX3i stack(4, 3);
  stack << 1, 1, 1,
           1, 1, 1,
           0, 1, 1,
           operator_i, 1, 0;
  
  Eigen::ArrayXXd x_0 = sample_agraph_1_values.x_vals;
  Eigen::ArrayXXd constants = sample_agraph_1_values.constants;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> res_and_gradient = 
    simplifyAndEvaluateWithDerivative(stack,
                                      x_0,
                                      constants,
                                      false);
  Eigen::ArrayXXd df_dc = res_and_gradient.second;
  ASSERT_TRUE(testutils::almost_equal(expected_derivative, df_dc));
}
INSTANTIATE_TEST_CASE_P(,AGraphBackend, ::testing::Range(0, N_OPS, 1));


TEST_F(AGraphBackend, evaluate) {
  Eigen::ArrayXXd y = evaluate(simple_stack, x, constants);
  Eigen::ArrayXXd y_true = x.col(0) * (constants[0] + constants[1] 
                          / x.col(1)) - x.col(0);
  ASSERT_TRUE(testutils::almost_equal(y, y_true));
}

TEST_F(AGraphBackend, evaluate_and_derivative) {
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> y_and_dy =
    evaluateWithDerivative(simple_stack, x, constants);
  Eigen::ArrayXXd y_true = x.col(0) * (constants[0] + constants[1] 
                          / x.col(1)) - x.col(0);
  Eigen::ArrayXXd dy_true = Eigen::ArrayXXd::Zero(3, 3);
  dy_true.col(0) = constants[0] + constants[1] / x.col(1) - 1.;
  dy_true.col(1) = - x.col(0) * constants[1] / x.col(1) / x.col(1);

  ASSERT_TRUE(testutils::almost_equal(y_and_dy.first, y_true));
  ASSERT_TRUE(testutils::almost_equal(y_and_dy.second, dy_true));
}

TEST_F(AGraphBackend, mask_evaluate) {
  Eigen::ArrayXXd y = evaluate(simple_stack, x, constants);
  Eigen::ArrayXXd y_simple = simplifyAndEvaluate(simple_stack, x, constants);
  ASSERT_TRUE(testutils::almost_equal(y, y_simple));
}

TEST_F(AGraphBackend, mask_evaluate_and_derivative) {
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> y_and_dy =
    evaluateWithDerivative(simple_stack, x, constants);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> y_and_dy_simple =
    simplifyAndEvaluateWithDerivative(simple_stack, x, constants);
  ASSERT_TRUE(testutils::almost_equal(y_and_dy.first, y_and_dy_simple.first));
  ASSERT_TRUE(testutils::almost_equal(y_and_dy.first, y_and_dy_simple.first));
}

TEST_F(AGraphBackend, get_utilized_commands) {
  std::vector<bool> used_commands = getUtilizedCommands(simple_stack);
  int num_used_commands = 0;
  for (auto const& command_is_used : used_commands) {
    if (command_is_used) {
      ++num_used_commands;
    }
  }
  ASSERT_EQ(num_used_commands, 8);
}
} // namespace
