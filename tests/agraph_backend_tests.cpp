#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "gtest/gtest.h"
#include "BingoCpp/acyclic_graph.h"
#include "BingoCpp/graph_manip.h"

namespace {

struct AGraphValues {
  Eigen::ArrayXXd x_vals;
  Eigen::VectorXd constants;
  AGraphValues() {}
  AGraphValues(Eigen::ArrayXXd &x, Eigen::VectorXd &c) : 
    x_vals(x), constants(c) {}
};

class AGraphBackend : public ::testing::TestWithParam<int> {
 public:
  const double TESTING_TOL = 1e-7;
  const double AGRAPH_VAL_START =-1;
  const double AGRAPH_VAL_END = 0;
  const int N_AGRAPH_VAL = 11;
  const int N_OPS = 13;

  AGraphValues sample_agraph_1_values;
  std::vector<Eigen::ArrayXXd> operator_evals_x0;
  std::vector<Eigen::ArrayXXd> operator_x_derivs;
  std::vector<Eigen::ArrayXXd> operator_c_derivs;

  virtual void SetUp() {
    sample_agraph_1_values = init_agraph_vals(AGRAPH_VAL_START,
                                              AGRAPH_VAL_END,
                                              N_AGRAPH_VAL);
    operator_evals_x0 = init_op_evals_x0(sample_agraph_1_values);
    operator_x_derivs = init_op_x_derivs(sample_agraph_1_values);
    operator_c_derivs = init_op_c_derivs(sample_agraph_1_values);
  }
  virtual void TearDown() {}

  bool almostEqual(const Eigen::ArrayXXd &array1, const Eigen::ArrayXXd &array2) {
    if (non_comparable_matrices(array1, array2))
      return false;

    int rows_array1= array1.rows();
    int cols_array1= array1.cols();
    Eigen::MatrixXd matrix_diff = Eigen::MatrixXd(rows_array1, cols_array1);
    for (int row = 0; row < rows_array1; row++) {
      for (int col = 0; col < cols_array1; col++) {
        matrix_diff(row, col) = difference(array1(row, col), array1(row, col));
      }
    }
    double frobenius_norm = matrix_diff.norm();
    return (frobenius_norm < TESTING_TOL ? true : false);
  }

 private:
  double difference(double val_1, double val_2) {
    if ((std::isnan(val_1) && std::isnan(val_2)) ||
        (val_1 == INFINITY && val_2 == INFINITY) ||
        (val_1 == -INFINITY && val_2 == -INFINITY)) {
      return 0;
    } else {
      return val_1 - val_2;
    }
  }

  bool non_comparable_matrices(const Eigen::ArrayXXd &array1,
                                const Eigen::ArrayXXd &array2) {
    int rows_array1 = array1.rows();
    int rows_array2= array2.rows();
    int cols_array1= array1.cols();
    int cols_array2= array2.cols();
    return (!(rows_array1>0) || 
            !(rows_array2>0) || 
            !(cols_array1> 0) || 
            !(cols_array2> 0) || 
            (rows_array1 != rows_array2) || 
            (cols_array1!= cols_array2));
  }

  AGraphValues init_agraph_vals(double begin, double end, int num_points) {
    Eigen::VectorXd constants = Eigen::VectorXd(2);
    constants << 10, 3.14;

    Eigen::ArrayXXd x_vals(num_points, 2);
    x_vals.col(0) = Eigen::ArrayXd::LinSpaced(num_points, begin, end);
    x_vals.col(1) = Eigen::ArrayXd::LinSpaced(num_points, end, -begin);

    return AGraphValues(x_vals, constants);
  }

  std::vector<Eigen::ArrayXXd> init_op_evals_x0(AGraphValues &sample_agraph_1_values) {
    Eigen::ArrayXXd x_0 = sample_agraph_1_values.x_vals.col(0);
    double constant = sample_agraph_1_values.constants[0];
    Eigen::ArrayXXd c_0 = constant * Eigen::ArrayXd::Ones(x_0.rows());

    std::vector<Eigen::ArrayXXd> op_evals_x0 = std::vector<Eigen::ArrayXXd>();
    op_evals_x0.push_back(x_0);
    op_evals_x0.push_back(c_0);
    op_evals_x0.push_back(x_0+x_0);
    op_evals_x0.push_back(x_0-x_0);
    op_evals_x0.push_back(x_0*x_0);
    op_evals_x0.push_back(x_0/x_0);
    op_evals_x0.push_back(x_0.sin());
    op_evals_x0.push_back(x_0.cos());
    op_evals_x0.push_back(x_0.exp());
    op_evals_x0.push_back(x_0.abs().log());
    op_evals_x0.push_back(x_0.abs().pow(x_0));
    op_evals_x0.push_back(x_0.abs());
    op_evals_x0.push_back(x_0.abs().sqrt());

    return op_evals_x0;
  }

  std::vector<Eigen::ArrayXXd> init_op_x_derivs(AGraphValues &sample_agraph_1_values) {
    Eigen::ArrayXXd x_0 = sample_agraph_1_values.x_vals.col(0);
    std::vector<Eigen::ArrayXXd> op_x_derivs = std::vector<Eigen::ArrayXXd>();
    int size = x_0.rows();

    auto last_nan = [](Eigen::ArrayXXd array) {
      array(array.rows() - 1, array.cols() -1) = std::nan("1");
      Eigen::ArrayXXd modified_array = array;
      return modified_array;
    };
    op_x_derivs.push_back(Eigen::ArrayXd::Ones(size));
    op_x_derivs.push_back(Eigen::ArrayXd::Zero(size));
    op_x_derivs.push_back(2.0  * Eigen::ArrayXd::Ones(size));
    op_x_derivs.push_back(Eigen::ArrayXd::Zero(size));
    op_x_derivs.push_back(2.0 * x_0);
    op_x_derivs.push_back(last_nan(Eigen::ArrayXd::Zero(size)));
    op_x_derivs.push_back(x_0.cos());
    op_x_derivs.push_back(-x_0.sin());
    op_x_derivs.push_back(x_0.exp());
    op_x_derivs.push_back(1.0 / x_0);
    op_x_derivs.push_back(last_nan(x_0.abs().pow(x_0)*(x_0.abs().log()
                         + Eigen::ArrayXd::Ones(size))));
    op_x_derivs.push_back(x_0.sign());
    op_x_derivs.push_back(0.5 * x_0.sign() / x_0.abs().sqrt());

    return op_x_derivs;
  }

  std::vector<Eigen::ArrayXXd> init_op_c_derivs(AGraphValues &sample_agraph_1_values) {
    int size = sample_agraph_1_values.x_vals.rows();
    Eigen::ArrayXXd c_1 = sample_agraph_1_values.constants[1] * Eigen::ArrayXd::Ones(size);
    std::vector<Eigen::ArrayXXd> op_c_derivs = std::vector<Eigen::ArrayXXd>();

    op_c_derivs.push_back(Eigen::ArrayXd::Zero(size));
    op_c_derivs.push_back(Eigen::ArrayXd::Ones(size));
    op_c_derivs.push_back(2.0  * Eigen::ArrayXd::Ones(size));
    op_c_derivs.push_back(Eigen::ArrayXd::Zero(size));
    op_c_derivs.push_back(2.0 * c_1);
    op_c_derivs.push_back((Eigen::ArrayXd::Zero(size)));
    op_c_derivs.push_back(c_1.cos());
    op_c_derivs.push_back(-c_1.sin());
    op_c_derivs.push_back(c_1.exp());
    op_c_derivs.push_back(1.0 / c_1);
    op_c_derivs.push_back(c_1.abs().pow(c_1)*(c_1.abs().log()
                          + Eigen::ArrayXd::Ones(size)));
    op_c_derivs.push_back(c_1.sign());
    op_c_derivs.push_back(0.5 * c_1.sign() / c_1.abs().sqrt());

    return op_c_derivs;
  }
};

TEST_P(AGraphBackend, simplify_and_evaluate) {
  int operator_i = GetParam();
  Eigen::ArrayXXd expected_outcome = operator_evals_x0[operator_i];

  Eigen::ArrayX3i stack(3, 3);
  stack << 0, 0, 0,
           0, 1, 0,
           operator_i, 0, 0;
  Eigen::ArrayXXd f_of_x = SimplifyAndEvaluate(stack,
                                               sample_agraph_1_values.x_vals,
                                               sample_agraph_1_values.constants);
  ASSERT_TRUE(almostEqual(expected_outcome, f_of_x));
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
    SimplifyAndEvaluateWithDerivative(stack,
                                      x_0,
                                      constants,
                                      true);
  Eigen::ArrayXXd df_dx = res_and_gradient.second;
  ASSERT_TRUE(almostEqual(expected_derivative, df_dx));
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
    SimplifyAndEvaluateWithDerivative(stack,
                                      x_0,
                                      constants,
                                      false);
  Eigen::ArrayXXd df_dc = res_and_gradient.second;
  ASSERT_TRUE(almostEqual(expected_derivative, df_dc));
}


INSTANTIATE_TEST_CASE_P(,AGraphBackend, ::testing::Range(0, 13, 1));

} // namespace
