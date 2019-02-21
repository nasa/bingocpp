#ifndef INCLUDE_BINGOCPP_BACKEND_NODES_H_
#define INCLUDE_BINGOCPP_BACKEND_NODES_H_

#include <vector>
#include "acyclic_graph.h"

namespace backendnodes {
  typedef void (
    *forward_operator_function)(
      const Eigen::ArrayX3i&, const Eigen::ArrayXXd&,
      const Eigen::VectorXd&, std::vector<Eigen::ArrayXXd>&, std::size_t
  );
  typedef void (
    *derivative_operator_function)(
      const Eigen::ArrayX3i &, const int,
      const std::vector<Eigen::ArrayXXd> &,
      std::vector<Eigen::ArrayXXd> &, int
  );

  inline
  namespace { 
    void x_load_evaluate(const Eigen::ArrayX3i &stack,
                          const Eigen::ArrayXXd &x,
                          const Eigen::VectorXd &constants,
                          std::vector<Eigen::ArrayXXd> &buffer,
                          std::size_t result_location) {
      buffer[result_location] = x.col(stack(result_location, 1));
    }

    void x_load_deriv_evaluate(const Eigen::ArrayX3i &stack,
                                const int command_index,
                                const std::vector<Eigen::ArrayXXd> &forward_buffer,
                                std::vector<Eigen::ArrayXXd> &reverse_buffer,
                                int dependency) {
    }


    void c_load_evaluate(const Eigen::ArrayX3i &stack,
                          const Eigen::ArrayXXd &x,
                          const Eigen::VectorXd &constants,
                          std::vector<Eigen::ArrayXXd> &buffer,
                          std::size_t result_location) {
      if (stack(result_location, 0) != -1) {
        buffer[result_location] = Eigen::ArrayXXd::Constant(x.rows(), 1,
                                  constants[stack(result_location, 1)]);

      } else {
        buffer[result_location] = Eigen::ArrayXXd::Zero(x.rows(), 1);
      }
    }

    void c_load_deriv_evaluate(const Eigen::ArrayX3i &stack,
                                const int command_index,
                                const std::vector<Eigen::ArrayXXd> &forward_buffer,
                                std::vector<Eigen::ArrayXXd> &reverse_buffer,
                                int dependency) {
    }


    void addition_evaluate(const Eigen::ArrayX3i &stack,
                            const Eigen::ArrayXXd &x,
                            const Eigen::VectorXd &constants,
                            std::vector<Eigen::ArrayXXd> &buffer,
                            std::size_t result_location) {
      buffer[result_location] = buffer[stack(result_location, 1)] +
                                buffer[stack(result_location, 2)];
    }

    void addition_deriv_evaluate(const Eigen::ArrayX3i &stack,
                                  const int command_index,
                                  const std::vector<Eigen::ArrayXXd> &forward_buffer,
                                  std::vector<Eigen::ArrayXXd> &reverse_buffer,
                                  int dependency) {
      if (stack(dependency, 1) == command_index) {
        reverse_buffer[command_index] += reverse_buffer[dependency];
      }

      if (stack(dependency, 2) == command_index) {
        reverse_buffer[command_index] += reverse_buffer[dependency];
      }
    }


    void subtraction_evaluate(const Eigen::ArrayX3i &stack,
                              const Eigen::ArrayXXd &x,
                              const Eigen::VectorXd &constants,
                              std::vector<Eigen::ArrayXXd> &buffer,
                              std::size_t result_location) {
      buffer[result_location] = buffer[stack(result_location, 1)] -
                                buffer[stack(result_location, 2)];
    }

    void subtraction_deriv_evaluate(const Eigen::ArrayX3i &stack,
                                    const int command_index,
                                    const std::vector<Eigen::ArrayXXd> &forward_buffer,
                                    std::vector<Eigen::ArrayXXd> &reverse_buffer,
                                    int dependency) {
      if (stack(dependency, 1) == command_index) {
        reverse_buffer[command_index] += reverse_buffer[dependency];
      }

      if (stack(dependency, 2) == command_index) {
        reverse_buffer[command_index] -= reverse_buffer[dependency];
      }
    }


    void multiplication_evaluate(const Eigen::ArrayX3i &stack,
                                  const Eigen::ArrayXXd &x,
                                  const Eigen::VectorXd &constants,
                                  std::vector<Eigen::ArrayXXd> &buffer,
                                  std::size_t result_location) {
      buffer[result_location] = buffer[stack(result_location, 1)] *
                                buffer[stack(result_location, 2)];
    }

    void multiplication_deriv_evaluate(const Eigen::ArrayX3i &stack,
                                        const int command_index,
                                        const std::vector<Eigen::ArrayXXd> &forward_buffer,
                                        std::vector<Eigen::ArrayXXd> &reverse_buffer,
                                        int dependency) {
      if (stack(dependency, 1) == command_index) {
        reverse_buffer[command_index] += reverse_buffer[dependency] *
                                        forward_buffer[stack(dependency, 2)];
      }

      if (stack(dependency, 2) == command_index) {
        reverse_buffer[command_index] += reverse_buffer[dependency] *
                                        forward_buffer[stack(dependency, 1)];
      }
    }

    void division_evaluate(const Eigen::ArrayX3i &stack,
                            const Eigen::ArrayXXd &x,
                            const Eigen::VectorXd &constants,
                            std::vector<Eigen::ArrayXXd> &buffer,
                            std::size_t result_location) {
      buffer[result_location] = buffer[stack(result_location, 1)] /
                                buffer[stack(result_location, 2)];
    }

    void division_deriv_evaluate(const Eigen::ArrayX3i &stack,
                                  const int command_index,
                                  const std::vector<Eigen::ArrayXXd> &forward_buffer,
                                  std::vector<Eigen::ArrayXXd> &reverse_buffer,
                                  int dependency) {
      if (stack(dependency, 1) == command_index) {
        reverse_buffer[command_index] += reverse_buffer[dependency] /
                                        forward_buffer[stack(dependency, 2)];
      }

      if (stack(dependency, 2) == command_index) {
        reverse_buffer[command_index] += reverse_buffer[dependency] *
                                        (-forward_buffer[dependency] /
                                          forward_buffer[stack(dependency, 2)]);
      }
    }


    void sin_evaluate(const Eigen::ArrayX3i &stack,
                      const Eigen::ArrayXXd &x,
                      const Eigen::VectorXd &constants,
                      std::vector<Eigen::ArrayXXd> &buffer,
                      std::size_t result_location) {
      buffer[result_location] = buffer[stack(result_location, 1)].sin();
    }

    void sin_deriv_evaluate(const Eigen::ArrayX3i &stack,
                            const int command_index,
                            const std::vector<Eigen::ArrayXXd> &forward_buffer,
                            std::vector<Eigen::ArrayXXd> &reverse_buffer,
                            int dependency) {
      reverse_buffer[command_index] += reverse_buffer[dependency] *
                                      forward_buffer[stack(dependency, 1)].cos();
    }


    void cos_evaluate(const Eigen::ArrayX3i &stack,
                      const Eigen::ArrayXXd &x,
                      const Eigen::VectorXd &constants,
                      std::vector<Eigen::ArrayXXd> &buffer,
                      std::size_t result_location) {
      buffer[result_location] = buffer[stack(result_location, 1)].cos();
    }

    void cos_deriv_evaluate(const Eigen::ArrayX3i &stack,
                            const int command_index,
                            const std::vector<Eigen::ArrayXXd> &forward_buffer,
                            std::vector<Eigen::ArrayXXd> &reverse_buffer,
                            int dependency) {
      reverse_buffer[command_index] -= reverse_buffer[dependency] *
                                      forward_buffer[stack(dependency, 1)].sin();
    }


    void exp_evaluate(const Eigen::ArrayX3i &stack,
                      const Eigen::ArrayXXd &x,
                      const Eigen::VectorXd &constants,
                      std::vector<Eigen::ArrayXXd> &buffer,
                      std::size_t result_location) {
      buffer[result_location] = buffer[stack(result_location, 1)].exp();
    }

    void exp_deriv_evaluate(const Eigen::ArrayX3i &stack,
                            const int command_index,
                            const std::vector<Eigen::ArrayXXd> &forward_buffer,
                            std::vector<Eigen::ArrayXXd> &reverse_buffer,
                            int dependency) {
      reverse_buffer[command_index] += reverse_buffer[dependency] *
                                      forward_buffer[dependency];
    }


    void log_evaluate(const Eigen::ArrayX3i &stack,
                      const Eigen::ArrayXXd &x,
                      const Eigen::VectorXd &constants,
                      std::vector<Eigen::ArrayXXd> &buffer,
                      std::size_t result_location) {
      buffer[result_location] = (buffer[stack(result_location, 1)].abs()).log();
    }

    void log_deriv_evaluate(const Eigen::ArrayX3i &stack,
                            const int command_index,
                            const std::vector<Eigen::ArrayXXd> &forward_buffer,
                            std::vector<Eigen::ArrayXXd> &reverse_buffer,
                            int dependency) {
      reverse_buffer[command_index] += reverse_buffer[dependency] /
                                      forward_buffer[stack(dependency, 1)];
    }

    void power_evaluate(const Eigen::ArrayX3i &stack,
                        const Eigen::ArrayXXd &x,
                        const Eigen::VectorXd &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location) {
      buffer[result_location] = (buffer[stack(result_location, 1)].abs()).pow(
                                  buffer[stack(result_location, 2)]);
    }

    void power_deriv_evaluate(const Eigen::ArrayX3i &stack,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency) {
      if (stack(dependency, 1) == command_index) {
        reverse_buffer[command_index] += forward_buffer[dependency] *
                                        reverse_buffer[dependency] *
                                        forward_buffer[stack(dependency, 2)] /
                                        forward_buffer[stack(dependency, 1)];
      }

      if (stack(dependency, 2) == command_index) {
        reverse_buffer[command_index] += forward_buffer[dependency] *
                                        reverse_buffer[dependency] *
                                        (forward_buffer[stack(dependency, 1)].abs()).log();
      }
    }


    void absolute_evaluate(const Eigen::ArrayX3i &stack,
                            const Eigen::ArrayXXd &x,
                            const Eigen::VectorXd &constants,
                            std::vector<Eigen::ArrayXXd> &buffer,
                            std::size_t result_location) {
      buffer[result_location] = buffer[stack(result_location, 1)].abs();
    }

    void absolute_deriv_evaluate(const Eigen::ArrayX3i &stack,
                                  const int command_index,
                                  const std::vector<Eigen::ArrayXXd> &forward_buffer,
                                  std::vector<Eigen::ArrayXXd> &reverse_buffer,
                                  int dependency) {
      reverse_buffer[command_index] += reverse_buffer[dependency] *
                                      forward_buffer[stack(dependency, 1)].sign();
    }

    void sqrt_evaluate(const Eigen::ArrayX3i &stack,
                        const Eigen::ArrayXXd &x,
                        const Eigen::VectorXd &constants,
                        std::vector<Eigen::ArrayXXd> &buffer,
                        std::size_t result_location) {
      buffer[result_location] = (buffer[stack(result_location, 1)].abs()).sqrt();
    }

    void sqrt_deriv_evaluate(const Eigen::ArrayX3i &stack,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency) {
      reverse_buffer[command_index] += 0.5 * reverse_buffer[dependency] /
                                      forward_buffer[dependency] *
                                      forward_buffer[stack(dependency, 1)].sign();
    }
  } //namespace

  const std::vector<forward_operator_function> forward_eval_map {
    x_load_evaluate,
    c_load_evaluate,
    addition_evaluate,
    subtraction_evaluate,
    multiplication_evaluate,
    division_evaluate,
    sin_evaluate,
    cos_evaluate,
    exp_evaluate,
    log_evaluate,
    power_evaluate,
    absolute_evaluate,
    sqrt_evaluate
  };

  const std::vector<derivative_operator_function> derivative_eval_map {
    x_load_deriv_evaluate,
    c_load_deriv_evaluate,
    addition_deriv_evaluate,
    subtraction_deriv_evaluate,
    multiplication_deriv_evaluate,
    division_deriv_evaluate,
    sin_deriv_evaluate,
    cos_deriv_evaluate,
    exp_deriv_evaluate,
    log_deriv_evaluate,
    power_deriv_evaluate,
    absolute_deriv_evaluate,
    sqrt_deriv_evaluate
  };

  inline 
  void forward_eval_function(int node, const Eigen::ArrayX3i &stack,
                                        const Eigen::ArrayXXd &x,
                                        const Eigen::VectorXd &constants,
                                        std::vector<Eigen::ArrayXXd> &buffer,
                                        std::size_t result_location) {
    forward_eval_map.at(node)(stack, x, constants, buffer, result_location);
  }

  inline
  void derivative_eval_function(int node, const Eigen::ArrayX3i &stack,
                              const int command_index,
                              const std::vector<Eigen::ArrayXXd> &forward_buffer,
                              std::vector<Eigen::ArrayXXd> &reverse_buffer,
                              int dependency) {
    derivative_eval_map.at(node)(stack, command_index, forward_buffer, reverse_buffer, dependency);
  }
} //backendnodes

#endif