#ifndef INCLUDE_BINGOCPP_BACKEND_NODES_H_
#define INCLUDE_BINGOCPP_BACKEND_NODES_H_

#include <vector>
#include <Eigen/Dense>

namespace backendnodes {
  typedef Eigen::ArrayXXd (
    *forward_operator_function)(
      int, int, const Eigen::ArrayXXd&, const Eigen::VectorXd&, Eigen::ArrayXXd&
  );
  typedef void (
    *reverse_operator_function)(
      int, int, int, const Eigen::ArrayXXd&, Eigen::ArrayXXd&
  );
  
  Eigen::ArrayXXd forward_eval_function(int node, int param1, int param2,
                                         const Eigen::ArrayXXd &x, 
                                         const Eigen::VectorXd &constants,
                                         Eigen::ArrayXXd &forward_eval);

  void reverse_eval_function(int node, int reverse_index, int param1, int param2,
                                        const Eigen::ArrayXXd &forward_eval,
                                        Eigen::ArrayXXd &reverse_eval);
} //backendnodes

#endif