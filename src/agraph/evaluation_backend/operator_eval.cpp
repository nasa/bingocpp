#include <bingocpp/agraph/evaluation_backend/operator_eval.h>

namespace bingo {
namespace evaluation_backend {
namespace { 

// Load x
Eigen::ArrayXXd loadx_forward_eval(int param1, int, 
                                   const Eigen::ArrayXXd &x, 
                                   const Eigen::VectorXd &, 
                                   std::vector<Eigen::ArrayXXd> &) {
  return x.col(param1);
}

void loadx_reverse_eval(int, int, int,
                        const std::vector<Eigen::ArrayXXd> &,
                        std::vector<Eigen::ArrayXXd> &) {
  return;
}

// Load c
Eigen::ArrayXXd loadc_forward_eval(int param1, int, 
                                   const Eigen::ArrayXXd &x, 
                                   const Eigen::VectorXd &constants, 
                                   std::vector<Eigen::ArrayXXd> &) {
  return Eigen::ArrayXd::Constant(x.rows(), constants[param1]);
}

void loadc_reverse_eval(int, int, int,
                        const std::vector<Eigen::ArrayXXd> &,
                        std::vector<Eigen::ArrayXXd> &) {
  return;
}

// Addition
Eigen::ArrayXXd add_forward_eval(int param1, int param2, 
                                 const Eigen::ArrayXXd &, 
                                 const Eigen::VectorXd &, 
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1] + forward_eval[param2]; 
} 

void add_reverse_eval(int reverse_index, int param1, int param2, 
                      const std::vector<Eigen::ArrayXXd> &, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index];
  reverse_eval[param2] += reverse_eval[reverse_index];
} 

// Subtraction
Eigen::ArrayXXd subtract_forward_eval(int param1, int param2, 
                                      const Eigen::ArrayXXd &,
                                      const Eigen::VectorXd &, 
                                      std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1] - forward_eval[param2]; 
} 

void subtract_reverse_eval(int reverse_index, int param1, int param2, 
                           const std::vector<Eigen::ArrayXXd> &, 
                           std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index];
  reverse_eval[param2] -= reverse_eval[reverse_index];
}

// Multiplication
Eigen::ArrayXXd multiply_forward_eval(int param1, int param2, 
                                      const Eigen::ArrayXXd &, 
                                      const Eigen::VectorXd &, 
                                      std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1] * forward_eval[param2]; 
} 

void multiply_reverse_eval(int reverse_index, int param1, int param2, 
                           const std::vector<Eigen::ArrayXXd> &forward_eval, 
                           std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                              *forward_eval[param2];
  reverse_eval[param2] += reverse_eval[reverse_index]
                              *forward_eval[param1];
} 

// Division
Eigen::ArrayXXd divide_forward_eval(int param1, int param2, 
                                    const Eigen::ArrayXXd &, 
                                    const Eigen::VectorXd &, 
                                    std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1] / forward_eval[param2]; 
} 

void divide_reverse_eval(int reverse_index, int param1, int param2, 
                         const std::vector<Eigen::ArrayXXd> &forward_eval, 
                         std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                              /forward_eval[param2];
  reverse_eval[param2] -= reverse_eval[reverse_index]
                              *forward_eval[reverse_index]
                              /forward_eval[param2];
}

// Sine
Eigen::ArrayXXd sin_forward_eval(int param1, int, 
                                 const Eigen::ArrayXXd &, 
                                 const Eigen::VectorXd &, 
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval.at(param1).sin(); 
}

void sin_reverse_eval(int reverse_index, int param1, int, 
                      const std::vector<Eigen::ArrayXXd> &forward_eval, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                         *forward_eval[param1].cos();
}

// Cosine
Eigen::ArrayXXd cos_forward_eval(int param1, int, 
                                 const Eigen::ArrayXXd &, 
                                 const Eigen::VectorXd &, 
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1].cos(); 
}

void cos_reverse_eval(int reverse_index, int param1, int, 
                      const std::vector<Eigen::ArrayXXd> &forward_eval, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] -= reverse_eval[reverse_index]
                         *forward_eval[param1].sin();
}

// Exponential 
Eigen::ArrayXXd exp_forward_eval(int param1, int,
                                 const Eigen::ArrayXXd &,
                                 const Eigen::VectorXd &,
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1].exp();
}

void exp_reverse_eval(int reverse_index, int param1, int, 
                      const std::vector<Eigen::ArrayXXd> &forward_eval, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                         *forward_eval[reverse_index];
}

// Logarithm
Eigen::ArrayXXd log_forward_eval(int param1, int,
                                 const Eigen::ArrayXXd &,
                                 const Eigen::VectorXd &,
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1].abs().log();
}

void log_reverse_eval(int reverse_index, int param1, int, 
                      const std::vector<Eigen::ArrayXXd> &forward_eval, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                         /forward_eval[param1];
}

// Power
Eigen::ArrayXXd pow_forward_eval(int param1, int param2,
                                 const Eigen::ArrayXXd &,
                                 const Eigen::VectorXd &,
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1].abs().pow(forward_eval[param2]);
}

void pow_reverse_eval(int reverse_index, int param1, int param2, 
                      const std::vector<Eigen::ArrayXXd> &forward_eval, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                         *forward_eval[reverse_index]
                         *forward_eval[param2]
                         /forward_eval[param1];
  reverse_eval[param2] += reverse_eval[reverse_index]
                         *forward_eval[reverse_index]
                         *(forward_eval[param1].abs().log());
}

// Absolute Value
Eigen::ArrayXXd abs_forward_eval(int param1, int,
                                 const Eigen::ArrayXXd &,
                                 const Eigen::VectorXd &,
                                 std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1].abs();
}

void abs_reverse_eval(int reverse_index, int param1, int, 
                      const std::vector<Eigen::ArrayXXd> &forward_eval, 
                      std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += reverse_eval[reverse_index]
                         *forward_eval[param1].sign();
}

// Sqruare root
Eigen::ArrayXXd sqrt_forward_eval(int param1, int,
                                  const Eigen::ArrayXXd &,
                                  const Eigen::VectorXd &,
                                  std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval[param1].abs().sqrt();
}

void sqrt_reverse_eval(int reverse_index, int param1, int, 
                       const std::vector<Eigen::ArrayXXd> &forward_eval, 
                       std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval[param1] += 0.5*reverse_eval[reverse_index]
                              /forward_eval[reverse_index]
                              *forward_eval[param1].sign();
}

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
  subtract_reverse_eval,
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
} // namespace

Eigen::ArrayXXd ForwardEvalFunction(int node, int param1, int param2,
                                    const Eigen::ArrayXXd &x, 
                                    const Eigen::VectorXd &constants,
                                    std::vector<Eigen::ArrayXXd> &forward_eval) {
  return forward_eval_map.at(node)(param1, param2, x, constants, forward_eval);
}

void ReverseEvalFunction(int node, int reverse_index, int param1, int param2,
                         const std::vector<Eigen::ArrayXXd> &forward_eval,
                         std::vector<Eigen::ArrayXXd> &reverse_eval) {
  reverse_eval_map.at(node)(reverse_index, param1, param2, forward_eval, reverse_eval);
}
} // namespace backend
} // namespace bingo