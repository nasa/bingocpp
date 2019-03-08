#ifndef BINGO_TESTS_TEST_FIXTURES_H_
#define BINGO_TESTS_TEST_FIXTURES_H_

#include "testing_utils.h"
#include "BingoCpp/acyclic_graph.h"
#include "BingoCpp/graph_manip.h"

namespace testutils {

inline Eigen::ArrayXXd one_to_nine_3_by_3() {
  Eigen::ArrayXXd x_vals(3, 3);
  x_vals << 1., 4., 7., 2., 5., 8., 3., 6., 9.;
  return x_vals;
}

inline Eigen::ArrayX3i stack_unary_operator(int op, int param=0) {
  Eigen::ArrayX3i stack(2, 3);
  stack << param, 0, 0,
           op, 0, 0;
  return stack;
}

inline Eigen::ArrayX3i stack_binary_operator(int op, int operand1=0, 
                                             int operand2=1) {
  Eigen::ArrayX3i stack(3,3);
  stack << 0, 0, 0,
           1, 0, 0,
           op, operand1, operand2;
  return stack;
}

// y = x_0 * ( C_0 + C_1/x_1 ) - x_0
inline Eigen::ArrayX3i stack_operators_0_to_5() {
  Eigen::ArrayX3i stack(12, 3);
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
  return stack;
}

inline Eigen::VectorXd pi_ten_constants() {
  Eigen::VectorXd constants(2);
  constants << 3.14, 10;
  return constants;
} 

} // namespace testutils
#endif //BINGO_TESTS_TEST_FIXTURES_H_