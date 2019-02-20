#include <cmath>
#include <vector>

#include <Eigen/Dense>

#include "gtest/gtest.h"
#include "BingoCpp/backend_nodes.h"
#include "testing_utils.h"

namespace {

class BackendNodes : public ::testing::TestWithParam<int>{
 public:
  const double X_START =-1;
  const double X_END = 0;
  const int NUM_X_VALS = 11;

  const int X_IDX = 0;

  testutils::AGraphValues sample_agraph_data;
  Eigen::ArrayXXd forward_eval_x0;
  Eigen::ArrayXXd reverse_eval;

  virtual void SetUp() {
    sample_agraph_data = testutils::init_agraph_vals(X_START, X_END, NUM_X_VALS);
    init_forward_eval(forward_eval_x0, sample_agraph_data);
  }
  virtual void TearDown() {}
 private:
  void init_forward_eval(Eigen::ArrayXXd &forward_eval_x0,  testutils::AGraphValues &sample_agraph_data) {
    std::vector<Eigen::ArrayXXd> vec_eval = testutils::init_op_evals_x0(sample_agraph_data);
    forward_eval_x0 = Eigen::ArrayXXd::Zero(vec_eval.size(), sample_agraph_data.x_vals.rows());
    for (int i=0; i<vec_eval.size(); i++) {
      forward_eval_x0.row(i) = vec_eval[i].transpose();
    }
  }
};

TEST_P(BackendNodes, forward_eval) {
  int node_op = GetParam();
  Eigen::ArrayXXd res = backendnodes::forward_eval_function(
    node_op, X_IDX, X_IDX, 
    sample_agraph_data.x_vals, 
    sample_agraph_data.constants, 
    forward_eval_x0);
  ASSERT_TRUE(testutils::almost_equal(forward_eval_x0.row(node_op), res));
}

// TEST_P(BackendNodes, reverse_eval) {
//   int node_op = GetParam();
//   backendnodes::reverse_eval_function(node_op, X_IDX, X_IDX, 
//     forward_eval_x0);
//   ASSERT_TRUE(testutils::almost_equal(forward_eval_x0.row(node_op), res));
// }

INSTANTIATE_TEST_CASE_P(,BackendNodes, ::testing::Range(0, 13, 1));
}// namespace