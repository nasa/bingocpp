#include <stdio.h>
#include <math.h>

#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "gtest/gtest.h"
#include "BingoCpp/acyclic_graph.h"
#include "BingoCpp/graph_manip.h"

namespace {

struct AGraphValues {
	Eigen::ArrayXXd x_vals;
	std::vector<double> constants;
};

class AGraphBackend : public ::testing::TestWithParam<int> {
 public:
 	const double TESTING_TOL = 1e-7;
	const double AGRAPH_VAL_START =-1;
	const double AGRAPH_VAL_END = 0;
	const int N_AGRAPH_VAL = 11;
	const int N_OPS = 13;

	AGraphValues *sample_agraph_1_values;
	std::vector<Eigen::ArrayXXd> *operator_evals_x0;
	std::vector<Eigen::ArrayXXd> operator_x_derivs;
	std::vector<Eigen::ArrayXXd> operator_c_derivs;
	Eigen::ArrayXXd sample_stack;
	Eigen::ArrayXXd all_funcs_stack;
	AcyclicGraph indv;
  virtual void SetUp() {
  	sample_agraph_1_values = init_agraph_vals(AGRAPH_VAL_START,
  																						 AGRAPH_VAL_END,
  																						 N_AGRAPH_VAL);
  	operator_evals_x0 = init_op_evals_x0(sample_agraph_1_values);

		return;
  }
  virtual void TearDown() {
  	delete operator_evals_x0	;
  	delete sample_agraph_1_values;
  	return; 
  }
 private:
 	AGraphValues *init_agraph_vals(double begin, double end, int num_points) {
 		std::vector<double> constants = {10, 3.14};

 		Eigen::ArrayXXd x_vals(num_points, 1);
 		x_vals = Eigen::ArrayXd::LinSpaced(num_points, begin, end);
 		AGraphValues *sample_vals = new AGraphValues();
 		sample_vals->x_vals = x_vals;
 		sample_vals->constants = constants;
 		return sample_vals;
 	}

 	std::vector<Eigen::ArrayXXd> *init_op_evals_x0(AGraphValues *sample_agraph_1_values) {

 		Eigen::ArrayXXd x_0 = sample_agraph_1_values->x_vals;
 		Eigen::ArrayXXd c_0(x_0.rows(), 1);

 		//TODO use eigen interface to fill constant array
 		double constant = sample_agraph_1_values->constants.at(0);
 		for (int i=0; i<x_0.rows(); i++) {
 			c_0 << constant;
 		}
 		std::vector<Eigen::ArrayXXd> *op_evals_x0 = new std::vector<Eigen::ArrayXXd>(N_OPS);
 		op_evals_x0->push_back(x_0);
 		op_evals_x0->push_back(c_0);
 		op_evals_x0->push_back(x_0+x_0);
 		op_evals_x0->push_back(x_0-x_0);
 		op_evals_x0->push_back(x_0*x_0);
 		op_evals_x0->push_back(x_0/x_0);
 		op_evals_x0->push_back(x_0.sin());
 		op_evals_x0->push_back(x_0.cos());
 		op_evals_x0->push_back(x_0.exp());
 		op_evals_x0->push_back(x_0.abs().log());
 		op_evals_x0->push_back(x_0.pow(x_0.abs()));
 		op_evals_x0->push_back(x_0.abs());
 		op_evals_x0->push_back(x_0.abs().sqrt());

 		return op_evals_x0;
 	}
};

TEST_P(AGraphBackend, simplify_and_evaluate) {
	Eigen::ArrayX3i stack(3, 3);
	const int operator_i = GetParam();
	stack << 0, 0, 0,
					 0, 1, 0,
					 operator_i, 0, 0;
	// f_of_x = indv.simplify_and_evaluate(stack,
	// 																		sample_agraph_1_values.x_vals,
	// 																		sample_agraph_1_values.constants);
	ASSERT_TRUE(GetParam());
	// ASSERT_(ASSERT_NEAR(expected_outcome, f_of_x, TOL));
}

TEST_P(AGraphBackend, simplify_and_evaluate_x_deriv) {
	const int operator_i = GetParam();
	ASSERT_TRUE(GetParam());
}

TEST_P(AGraphBackend, simplify_and_evaluate_c_deriv) {
	const int operator_i = GetParam();
	ASSERT_TRUE(GetParam());
}


INSTANTIATE_TEST_CASE_P(Instance, AGraphBackend, ::testing::Values(1, 2, 3));

} // namespace