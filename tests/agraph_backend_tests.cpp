#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

#include "gtest/gtest.h"
#include "BingoCpp/acyclic_graph.h"
#include "BingoCpp/graph_manip.h"

// TODO refactor code to work with ArrayXd instead of ArrayXXd
namespace {

struct AGraphValues {
	Eigen::ArrayXXd x_vals;
	Eigen::VectorXd constants;
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

 private:
 	AGraphValues init_agraph_vals(double begin, double end, int num_points) {
 		Eigen::VectorXd constants = Eigen::VectorXd(2);
 		constants << 10, 3.14;

 		Eigen::ArrayXXd x_vals(num_points, 1);
 		x_vals = Eigen::ArrayXd::LinSpaced(num_points, begin, end);
 		AGraphValues sample_vals = AGraphValues();
 		sample_vals.x_vals = x_vals;
 		sample_vals.constants = constants;
 		return sample_vals;
 	}

 	std::vector<Eigen::ArrayXXd> init_op_evals_x0(AGraphValues &sample_agraph_1_values) {

 		Eigen::ArrayXXd x_0 = sample_agraph_1_values.x_vals;

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

 		Eigen::ArrayXXd x_0 = sample_agraph_1_values.x_vals;
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
 		op_x_derivs.push_back(last_nan(x_0.abs().pow(x_0)*(x_0.abs().log() + Eigen::ArrayXd::Ones(size))));
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
 		op_c_derivs.push_back(c_1.abs().pow(c_1)*(c_1.abs().log() + Eigen::ArrayXd::Ones(size)));
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

	ASSERT_TRUE(Eigen::MatrixXd(expected_outcome).isApprox(Eigen::MatrixXd(f_of_x)));
}

TEST_P(AGraphBackend, simplify_and_evaluate_x_deriv) {
	const int operator_i = GetParam();
	ASSERT_TRUE(GetParam());
}

TEST_P(AGraphBackend, simplify_and_evaluate_c_deriv) {
	const int operator_i = GetParam();
	ASSERT_TRUE(GetParam());
}


INSTANTIATE_TEST_CASE_P(Instance, AGraphBackend, ::testing::Range(0, 13, 1));

} // namespace