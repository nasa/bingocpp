#ifndef INCLUDE_BINGOCPP_BACKEND_NODES_H_
#define INCLUDE_BINGOCPP_BACKEND_NODES_H_

#include <vector>
#include "acyclic_graph.h"

namespace backendnodes {
  typedef Eigen::ArrayXXd (
    *forward_operator_function)(
      int, int, const Eigen::ArrayXXd&, const Eigen::VectorXd&, Eigen::ArrayXXd&
  );
  typedef Eigen::ArrayXXd (
    *reverse_operator_function)(
      int, int, int, const Eigen::ArrayXXd&, Eigen::ArrayXXd&
  );
  
  Eigen::ArrayXXd loadx_forward_eval(int param1, int param2, 
                                     const Eigen::ArrayXXd &x, 
                                     const Eigen::VectorXd &constants, 
                                     Eigen::ArrayXXd &forward_eval) {
    return x.col(param1);
  }

  Eigen::ArrayXXd loadx_reverse_eval(int reverse_index, int param1, int param2,
                                     const Eigen::ArrayXXd &forward_eval,
                                     Eigen::ArrayXXd &reverse_eval) {
    return Eigen::ArrayXXd();
  }

  std::vector<forward_operator_function> forward_eval_map {
    loadx_forward_eval
  };
  std::vector<reverse_operator_function> reverse_eval_map {
    loadx_reverse_eval
  };
} //backendnodes

#endif