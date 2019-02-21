#ifndef INCLUDE_BINGOCPP_BACKEND_NODES_H_
#define INCLUDE_BINGOCPP_BACKEND_NODES_H_

#include <vector>
#include "acyclic_graph.h"

namespace backendnodes {
  typedef Eigen::ArrayXXd (
    *forward_operator_function)(
      int, int, const Eigen::ArrayXXd&, const Eigen::VectorXd&, Eigen::ArrayXXd&
  );
  typedef void (
    *reverse_operator_function)(
      int, int, int, const Eigen::ArrayXXd&, Eigen::ArrayXXd&
  );

  inline
  namespace { 
    Eigen::ArrayXXd loadx_forward_eval(int param1, int param2, 
                                      const Eigen::ArrayXXd &x, 
                                      const Eigen::VectorXd &constants, 
                                      Eigen::ArrayXXd &forward_eval) {
      return x.col(param1).transpose();
    }
    void loadx_reverse_eval(int reverse_index, int param1, int param2,
                                      const Eigen::ArrayXXd &forward_eval,
                                      Eigen::ArrayXXd &reverse_eval) {
      return;
    }

    Eigen::ArrayXXd loadc_forward_eval(int param1, int param2, 
                                      const Eigen::ArrayXXd &x, 
                                      const Eigen::VectorXd &constants, 
                                      Eigen::ArrayXXd &forward_eval) {
      return (Eigen::ArrayXd::Ones(x.rows()) * constants[param1]).transpose();
    }
    void loadc_reverse_eval(int reverse_index, int param1, int param2,
                                      const Eigen::ArrayXXd &forward_eval,
                                      Eigen::ArrayXXd &reverse_eval) {
      return;
    }

    Eigen::ArrayXXd add_forward_eval(int param1, int param2, 
                                    const Eigen::ArrayXXd &x, 
                                    const Eigen::VectorXd &constants, 
                                    Eigen::ArrayXXd &forward_eval) {
      return forward_eval.row(param1) + forward_eval.row(param2); 
    } 
    void add_reverse_eval(int reverse_index, int param1, int param2, 
                          const Eigen::ArrayXXd &forward_eval, 
                          Eigen::ArrayXXd &reverse_eval) {
      reverse_eval.row(param1) += reverse_eval.row(reverse_index);
      reverse_eval.row(param2) += reverse_eval.row(reverse_index);
    } 

    Eigen::ArrayXXd subtract_forward_eval(int param1, int param2, 
                                          const Eigen::ArrayXXd &x,
                                          const Eigen::VectorXd &constants, 
                                          Eigen::ArrayXXd &forward_eval) {
      return forward_eval.row(param1) - forward_eval.row(param2); 
    } 
    void subtract_forward_eval(int reverse_index, int param1, int param2, 
                              const Eigen::ArrayXXd &forward_eval, 
                              Eigen::ArrayXXd &reverse_eval) {
      reverse_eval.row(param1) += reverse_eval.row(reverse_index);
      reverse_eval.row(param2) -= reverse_eval.row(reverse_index);
    }

    Eigen::ArrayXXd multiply_forward_eval(int param1, int param2, 
                                    const Eigen::ArrayXXd &x, 
                                    const Eigen::VectorXd &constants, 
                                    Eigen::ArrayXXd &forward_eval) {
      return forward_eval.row(param1) * forward_eval.row(param2); 
    } 
    void multiply_reverse_eval(int reverse_index, int param1, int param2, 
                              const Eigen::ArrayXXd &forward_eval, 
                              Eigen::ArrayXXd &reverse_eval) {
      reverse_eval.row(param1) += reverse_eval.row(reverse_index)*forward_eval.row(param2);
      reverse_eval.row(param2) += reverse_eval.row(reverse_index)*forward_eval.row(param1);
    } 

    Eigen::ArrayXXd divide_forward_eval(int param1, int param2, 
                                        const Eigen::ArrayXXd &x, 
                                        const Eigen::VectorXd &constants, 
                                        Eigen::ArrayXXd &forward_eval) {
      return forward_eval.row(param1) / forward_eval.row(param2); 
    } 
    void divide_reverse_eval(int reverse_index, int param1, int param2, 
                          const Eigen::ArrayXXd &forward_eval, 
                          Eigen::ArrayXXd &reverse_eval) {
      reverse_eval.row(param1) += reverse_eval.row(reverse_index)/forward_eval.row(param2);
      reverse_eval.row(param2) -= reverse_eval.row(reverse_index)*forward_eval.row(reverse_index)/forward_eval.row(param2);
    }

    Eigen::ArrayXXd sin_forward_eval(int param1, int param2, 
                                        const Eigen::ArrayXXd &x, 
                                        const Eigen::VectorXd &constants, 
                                        Eigen::ArrayXXd &forward_eval) {
      return forward_eval.row(param1).sin(); 
    } 
    void sin_reverse_eval(int reverse_index, int param1, int param2, 
                          const Eigen::ArrayXXd &forward_eval, 
                          Eigen::ArrayXXd &reverse_eval) {
      reverse_eval.row(param1) += reverse_eval.row(reverse_index)*forward_eval.row(param1).cos();
    }

    Eigen::ArrayXXd cos_forward_eval(int param1, int param2, 
                                        const Eigen::ArrayXXd &x, 
                                        const Eigen::VectorXd &constants, 
                                        Eigen::ArrayXXd &forward_eval) {
      return forward_eval.row(param1).cos(); 
    } 
    void cos_reverse_eval(int reverse_index, int param1, int param2, 
                          const Eigen::ArrayXXd &forward_eval, 
                          Eigen::ArrayXXd &reverse_eval) {
      reverse_eval.row(param1) -= reverse_eval.row(reverse_index)
                                  *forward_eval.row(param1).sin();
    }

    Eigen::ArrayXXd exp_forward_eval(int param1, int param2,
                                    const Eigen::ArrayXXd &x,
                                    const Eigen::VectorXd &constants,
                                    Eigen::ArrayXXd &forward_eval) {
      return forward_eval.row(param1).exp();
    }
    void exp_reverse_eval(int reverse_index, int param1, int param2, 
                          const Eigen::ArrayXXd &forward_eval, 
                          Eigen::ArrayXXd &reverse_eval) {
      reverse_eval.row(param1) += reverse_eval.row(reverse_index)
                                 *forward_eval.row(reverse_index);
    }

    Eigen::ArrayXXd log_forward_eval(int param1, int param2,
                                    const Eigen::ArrayXXd &x,
                                    const Eigen::VectorXd &constants,
                                    Eigen::ArrayXXd &forward_eval) {
      return forward_eval.row(param1).abs().log();
    }
    void log_reverse_eval(int reverse_index, int param1, int param2, 
                          const Eigen::ArrayXXd &forward_eval, 
                          Eigen::ArrayXXd &reverse_eval) {
      reverse_eval.row(param1) += reverse_eval.row(reverse_index)/forward_eval.row(param1);
    }

    Eigen::ArrayXXd pow_forward_eval(int param1, int param2,
                                    const Eigen::ArrayXXd &x,
                                    const Eigen::VectorXd &constants,
                                    Eigen::ArrayXXd &forward_eval) {
      return forward_eval.row(param1).abs().pow(forward_eval.row(param2));
    }
    void pow_reverse_eval(int reverse_index, int param1, int param2, 
                          const Eigen::ArrayXXd &forward_eval, 
                          Eigen::ArrayXXd &reverse_eval) {
      reverse_eval.row(param1) += reverse_eval.row(reverse_index)
                                 *forward_eval.row(reverse_index)
                                 *forward_eval.row(param2)
                                 /forward_eval.row(param1);
      reverse_eval.row(param1) += reverse_eval.row(reverse_index)
                                 *forward_eval.row(reverse_index)
                                 *forward_eval.row(param1).abs().log();
    }

    Eigen::ArrayXXd abs_forward_eval(int param1, int param2,
                                    const Eigen::ArrayXXd &x,
                                    const Eigen::VectorXd &constants,
                                    Eigen::ArrayXXd &forward_eval) {
      return forward_eval.row(param1).abs();
    }
    void abs_reverse_eval(int reverse_index, int param1, int param2, 
                          const Eigen::ArrayXXd &forward_eval, 
                          Eigen::ArrayXXd &reverse_eval) {
      reverse_eval.row(param1) += reverse_eval.row(reverse_index)
                                 *forward_eval.row(reverse_index).sign();
    }

    Eigen::ArrayXXd sqrt_forward_eval(int param1, int param2,
                                    const Eigen::ArrayXXd &x,
                                    const Eigen::VectorXd &constants,
                                    Eigen::ArrayXXd &forward_eval) {
      return forward_eval.row(param1).abs().sqrt();
    }
    void sqrt_reverse_eval(int reverse_index, int param1, int param2, 
                          const Eigen::ArrayXXd &forward_eval, 
                          Eigen::ArrayXXd &reverse_eval) {
      reverse_eval.row(param1) += 0.5*reverse_eval.row(reverse_index)
                                    *forward_eval.row(reverse_index).sign();
    }
  } //namespace

  const std::vector<forward_operator_function> forward_eval_map {
    loadx_forward_eval,
    loadc_forward_eval,
    add_forward_eval,
    subtract_forward_eval,
    multiply_forward_eval,
    divide_forward_eval,
    sin_forward_eval,
    cos_forward_eval,
    exp_forward_eval,
    log_forward_eval,
    pow_forward_eval,
    abs_forward_eval,
    sqrt_forward_eval
  };

  const std::vector<reverse_operator_function> reverse_eval_map {
    loadx_reverse_eval,
    loadc_reverse_eval,
    add_reverse_eval,
    multiply_reverse_eval,
    divide_reverse_eval,
    sin_reverse_eval,
    cos_reverse_eval,
    exp_reverse_eval,
    log_reverse_eval,
    pow_reverse_eval,
    abs_reverse_eval,
    sqrt_reverse_eval
  };

  inline 
  Eigen::ArrayXXd forward_eval_function(int node, int param1, int param2,
                                         const Eigen::ArrayXXd &x, 
                                         const Eigen::VectorXd &constants,
                                         Eigen::ArrayXXd &forward_eval) {
    return forward_eval_map.at(node)(param1, param2, x, constants, forward_eval);
  }

  inline
  void reverse_eval_function(int node, int reverse_index, int param1, int param2,
                                        const Eigen::ArrayXXd &forward_eval,
                                        Eigen::ArrayXXd &reverse_eval) {
    reverse_eval_map.at(node)(reverse_index, param1, param2, forward_eval, reverse_eval);
  }
} //backendnodes

#endif