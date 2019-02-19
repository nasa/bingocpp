#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "gtest/gtest.h"
#include "BingoCpp/backend_nodes.h"
#include "testing_utils.h"

namespace {

class BackendNodes : public ::testing::Test {
 public:
  const double X_START =0;
  const double X_END = 1;
  const int NUM_X_VALS = 11;
  const int X_OP = 0;
  const int C_OP = 1;
  const int ADD_OP = 2;
  const int SUB_OP = 3;
  const int MULT_OP = 4;
  const int DIV_OP = 5;
  const int SIN_OP = 6;
  const int COS_OP = 7;
  const int EXP_OP = 8;
  const int LOG_OP = 9;
  const int POW_OP = 10;
  const int ABS_OP = 11;
  const int SQRT_OP = 12;

  Eigen::ArrayXXd x_vals; 
  Eigen::ArrayX3i command_stack; // x*x + 2x
  Eigen::VectorXd constants;
  Eigen::ArrayXXd forward_eval;
  Eigen::ArrayXXd reverse_eval;

  virtual void SetUp() {
    init_x_vals(x_vals);
    init_stack(command_stack);
    init_constants(constants);
    forward_eval = Eigen::ArrayXXd(command_stack.rows(), x_vals.cols());
  }
  virtual void TearDown() {}
 private:
  void init_x_vals(Eigen::ArrayXXd &x_vals) {
    x_vals = Eigen::ArrayXXd(NUM_X_VALS, 2);
    x_vals.col(0) = Eigen::ArrayXd::LinSpaced(NUM_X_VALS, -1*X_END, X_START);
    x_vals.col(1) = Eigen::ArrayXd::LinSpaced(NUM_X_VALS, X_START, X_END);
  }
  void init_stack(Eigen::ArrayX3i &stack) {
    stack = Eigen::ArrayX3i(5, 3);
    stack << 0, 0, 0,
             1, 1, 1,
             4, 1, 0,
             4, 0, 0,
             2, 2, 3;
  }
  void init_constants(Eigen::VectorXd &constants) {
    constants = Eigen::ArrayXd(1);
    constants << 2.0;
  }
};

TEST_F(BackendNodes, loadx_forward_eval) {
  Eigen::ArrayXXd res = backendnodes::forward_eval_function(0, 0, 0, 
                                                            x_vals, 
                                                            constants, 
                                                            forward_eval);
  // ASSERT_TRUE(true);                                                            
  ASSERT_TRUE(testutils::almost_equal(x_vals.col(0), res));
}
}// namespace