#include <stdexcept>
#include <iostream>

#include <bingocpp/agraph/evaluation_backend/operator_eval.h>
#include <bingocpp/agraph/operator_definitions.h>

namespace bingo
{
  namespace evaluation_backend
  {
    namespace
    {
      //reshaping utility
      std::vector<Eigen::ArrayXXd> do_reshape(Eigen::ArrayXXd &buffer0, Eigen::ArrayXXd &buffer1)
      {
        Eigen::ArrayXXd tmp;
        std::vector<Eigen::ArrayXXd> reshaped_buffers(2);
        reshaped_buffers[0] = buffer0;
        reshaped_buffers[1] = buffer1;
        if (reshaped_buffers[0].rows() == reshaped_buffers[1].rows() 
            && reshaped_buffers[0].cols() == reshaped_buffers[1].cols()) {
          return reshaped_buffers;
        }

        if (reshaped_buffers[0].rows() == 1 && reshaped_buffers[1].rows() > 1) {
          tmp = reshaped_buffers[0].replicate(reshaped_buffers[1].rows(), 1);
          reshaped_buffers[0] = tmp;
        } else if (reshaped_buffers[1].rows() == 1 && reshaped_buffers[0].rows() > 1) {
          tmp = reshaped_buffers[1].replicate(reshaped_buffers[0].rows(), 1);
          reshaped_buffers[1] = tmp;
        } 

        if (reshaped_buffers[0].cols() == 1 && reshaped_buffers[1].cols() > 1) {
          tmp = reshaped_buffers[0].replicate(1, reshaped_buffers[1].cols());
          reshaped_buffers[0] = tmp;
        } else if (reshaped_buffers[1].cols() == 1 && reshaped_buffers[0].cols() > 1) {
          tmp = reshaped_buffers[1].replicate(1, reshaped_buffers[0].cols());
          reshaped_buffers[1] = tmp;
        } 

        return reshaped_buffers;
      }

      Eigen::ArrayXXd broadcast(const Eigen::ArrayXXd &array, int rows, int cols)
      {
          return array.replicate(rows / array.rows(), cols / array.cols());
      }


      // Integer
      Eigen::ArrayXXd integer_forward_eval(int param1, int,
                                           const Eigen::ArrayXXd &x,
                                           const Eigen::ArrayXXd &,
                                           std::vector<Eigen::ArrayXXd> &)
      {
        return Eigen::ArrayXXd::Constant(1, 1, param1);
      }

      void integer_reverse_eval(int, int, int,
                                const std::vector<Eigen::ArrayXXd> &,
                                std::vector<Eigen::ArrayXXd> &)
      {
        return;
      }

      // Load x
      Eigen::ArrayXXd loadx_forward_eval(int param1, int,
                                         const Eigen::ArrayXXd &x,
                                         const Eigen::ArrayXXd &constants,
                                         std::vector<Eigen::ArrayXXd> &)
      {
        return x.col(param1);
        // int num_cols = (constants.cols() == 0) ? 1 : constants.cols();
        // return x.col(param1).replicate(1, num_cols);
      }

      void loadx_reverse_eval(int, int, int,
                              const std::vector<Eigen::ArrayXXd> &,
                              std::vector<Eigen::ArrayXXd> &)
      {
        return;
      }

      // Load c
      Eigen::ArrayXXd loadc_forward_eval(int param1, int,
                                         const Eigen::ArrayXXd &x,
                                         const Eigen::ArrayXXd &constants,
                                         std::vector<Eigen::ArrayXXd> &)
      {
        // return Eigen::ArrayXXd::Constant(x.rows(), constants.columns(), constants(param1, 0));
        // return constants.row(param1).replicate(x.rows(), 1);
        return constants.row(param1);
      }

      void loadc_reverse_eval(int, int, int,
                              const std::vector<Eigen::ArrayXXd> &,
                              std::vector<Eigen::ArrayXXd> &)
      {
        return;
      }

      // Addition
      Eigen::ArrayXXd add_forward_eval(int param1, int param2,
                                       const Eigen::ArrayXXd &,
                                       const Eigen::ArrayXXd &,
                                       std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        std::vector<Eigen::ArrayXXd> reshaped_buffers = do_reshape(forward_eval[param1], forward_eval[param2]);
        return reshaped_buffers[0] + reshaped_buffers[1];
      }

      void add_reverse_eval(int reverse_index, int param1, int param2,
                            const std::vector<Eigen::ArrayXXd> &,
                            std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        reverse_eval[param1] += reverse_eval[reverse_index];
        reverse_eval[param2] += reverse_eval[reverse_index];
      }

      // Subtraction
      Eigen::ArrayXXd subtract_forward_eval(int param1, int param2,
                                            const Eigen::ArrayXXd &,
                                            const Eigen::ArrayXXd &,
                                            std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        std::vector<Eigen::ArrayXXd> reshaped_buffers = do_reshape(forward_eval[param1], forward_eval[param2]);
        return reshaped_buffers[0] - reshaped_buffers[1];
      }

      void subtract_reverse_eval(int reverse_index, int param1, int param2,
                                 const std::vector<Eigen::ArrayXXd> &,
                                 std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        reverse_eval[param1] += reverse_eval[reverse_index];
        reverse_eval[param2] -= reverse_eval[reverse_index];
      }

      // Multiplication
      Eigen::ArrayXXd multiply_forward_eval(int param1, int param2,
                                            const Eigen::ArrayXXd &,
                                            const Eigen::ArrayXXd &,
                                            std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        std::vector<Eigen::ArrayXXd> reshaped_buffers = do_reshape(forward_eval[param1], forward_eval[param2]);
        return reshaped_buffers[0] * reshaped_buffers[1];
      }

      void multiply_reverse_eval(int reverse_index, int param1, int param2,
                                 const std::vector<Eigen::ArrayXXd> &forward_eval,
                                 std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        Eigen::ArrayXXd fe1 = broadcast(forward_eval[param1], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        Eigen::ArrayXXd fe2 = broadcast(forward_eval[param2], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        reverse_eval[param1] += reverse_eval[reverse_index] * fe2;
        reverse_eval[param2] += reverse_eval[reverse_index] * fe1;
      }

      // Division
      Eigen::ArrayXXd divide_forward_eval(int param1, int param2,
                                          const Eigen::ArrayXXd &,
                                          const Eigen::ArrayXXd &,
                                          std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        std::vector<Eigen::ArrayXXd> reshaped_buffers = do_reshape(forward_eval[param1], forward_eval[param2]);
        return reshaped_buffers[0] / reshaped_buffers[1];
      }

      void divide_reverse_eval(int reverse_index, int param1, int param2,
                               const std::vector<Eigen::ArrayXXd> &forward_eval,
                               std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        Eigen::ArrayXXd fe2 = broadcast(forward_eval[param2], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        Eigen::ArrayXXd fer = broadcast(forward_eval[reverse_index], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        reverse_eval[param1] += reverse_eval[reverse_index] / fe2;
        reverse_eval[param2] -= reverse_eval[reverse_index] * fer / fe2;
      }

      // Sine
      Eigen::ArrayXXd sin_forward_eval(int param1, int,
                                       const Eigen::ArrayXXd &,
                                       const Eigen::ArrayXXd &,
                                       std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        return forward_eval.at(param1).sin();
      }

      void sin_reverse_eval(int reverse_index, int param1, int,
                            const std::vector<Eigen::ArrayXXd> &forward_eval,
                            std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        Eigen::ArrayXXd fe1 = broadcast(forward_eval[param1], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        reverse_eval[param1] += reverse_eval[reverse_index] * fe1.cos();
      }

      // Cosine
      Eigen::ArrayXXd cos_forward_eval(int param1, int,
                                       const Eigen::ArrayXXd &,
                                       const Eigen::ArrayXXd &,
                                       std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        return forward_eval[param1].cos();
      }

      void cos_reverse_eval(int reverse_index, int param1, int,
                            const std::vector<Eigen::ArrayXXd> &forward_eval,
                            std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        Eigen::ArrayXXd fe1 = broadcast(forward_eval[param1], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        reverse_eval[param1] -= reverse_eval[reverse_index] * fe1.sin();
      }

      // Exponential
      Eigen::ArrayXXd exp_forward_eval(int param1, int,
                                       const Eigen::ArrayXXd &,
                                       const Eigen::ArrayXXd &,
                                       std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        return forward_eval[param1].exp();
      }

      void exp_reverse_eval(int reverse_index, int param1, int,
                            const std::vector<Eigen::ArrayXXd> &forward_eval,
                            std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        Eigen::ArrayXXd fer = broadcast(forward_eval[reverse_index], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        reverse_eval[param1] += reverse_eval[reverse_index] * fer;
      }

      // Logarithm
      Eigen::ArrayXXd log_forward_eval(int param1, int,
                                       const Eigen::ArrayXXd &,
                                       const Eigen::ArrayXXd &,
                                       std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        return forward_eval[param1].abs().log();
      }

      void log_reverse_eval(int reverse_index, int param1, int,
                            const std::vector<Eigen::ArrayXXd> &forward_eval,
                            std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        Eigen::ArrayXXd fe1 = broadcast(forward_eval[param1], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        reverse_eval[param1] += reverse_eval[reverse_index] / fe1;
      }

      // Power
      Eigen::ArrayXXd pow_forward_eval(int param1, int param2,
                                       const Eigen::ArrayXXd &,
                                       const Eigen::ArrayXXd &,
                                       std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        std::vector<Eigen::ArrayXXd> reshaped_buffers = do_reshape(forward_eval[param1], forward_eval[param2]);
        return reshaped_buffers[0].pow(reshaped_buffers[1]);
      }

      void pow_reverse_eval(int reverse_index, int param1, int param2,
                            const std::vector<Eigen::ArrayXXd> &forward_eval,
                            std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        Eigen::ArrayXXd fe1 = broadcast(forward_eval[param1], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        Eigen::ArrayXXd fe2 = broadcast(forward_eval[param2], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        Eigen::ArrayXXd fer = broadcast(forward_eval[reverse_index], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        reverse_eval[param1] += reverse_eval[reverse_index] * fer * fe2 / fe1;
        reverse_eval[param2] += reverse_eval[reverse_index] * fer * (fe1.log());
      }

      // Safe Power
      Eigen::ArrayXXd safepow_forward_eval(int param1, int param2,
                                           const Eigen::ArrayXXd &,
                                           const Eigen::ArrayXXd &,
                                           std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        std::vector<Eigen::ArrayXXd> reshaped_buffers = do_reshape(forward_eval[param1], forward_eval[param2]);
        return reshaped_buffers[0].abs().pow(reshaped_buffers[1]);
      }

      void safepow_reverse_eval(int reverse_index, int param1, int param2,
                                const std::vector<Eigen::ArrayXXd> &forward_eval,
                                std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        Eigen::ArrayXXd fe1 = broadcast(forward_eval[param1], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        Eigen::ArrayXXd fe2 = broadcast(forward_eval[param2], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        Eigen::ArrayXXd fer = broadcast(forward_eval[reverse_index], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        reverse_eval[param1] += reverse_eval[reverse_index] * fer * fe2 / fe1;
        reverse_eval[param2] += reverse_eval[reverse_index] * fer * (fe1.abs().log());
      }

      // Absolute Value
      Eigen::ArrayXXd abs_forward_eval(int param1, int,
                                       const Eigen::ArrayXXd &,
                                       const Eigen::ArrayXXd &,
                                       std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        return forward_eval[param1].abs();
      }

      void abs_reverse_eval(int reverse_index, int param1, int,
                            const std::vector<Eigen::ArrayXXd> &forward_eval,
                            std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        Eigen::ArrayXXd fe1 = broadcast(forward_eval[param1], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        reverse_eval[param1] += reverse_eval[reverse_index] * fe1.sign();
      }

      // Sqruare root
      Eigen::ArrayXXd sqrt_forward_eval(int param1, int,
                                        const Eigen::ArrayXXd &,
                                        const Eigen::ArrayXXd &,
                                        std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        return forward_eval[param1].abs().sqrt();
      }

      void sqrt_reverse_eval(int reverse_index, int param1, int,
                             const std::vector<Eigen::ArrayXXd> &forward_eval,
                             std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        Eigen::ArrayXXd fe1 = broadcast(forward_eval[param1], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        Eigen::ArrayXXd fer = broadcast(forward_eval[reverse_index], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        reverse_eval[param1] += 0.5 * reverse_eval[reverse_index] / fer * fe1.sign();
      }

      // Sinh
      Eigen::ArrayXXd sinh_forward_eval(int param1, int,
                                        const Eigen::ArrayXXd &,
                                        const Eigen::ArrayXXd &,
                                        std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        return forward_eval.at(param1).sinh();
      }

      void sinh_reverse_eval(int reverse_index, int param1, int,
                             const std::vector<Eigen::ArrayXXd> &forward_eval,
                             std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        Eigen::ArrayXXd fe1 = broadcast(forward_eval[param1], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        reverse_eval[param1] += reverse_eval[reverse_index] * fe1.cosh();
      }

      // Cosh
      Eigen::ArrayXXd cosh_forward_eval(int param1, int,
                                        const Eigen::ArrayXXd &,
                                        const Eigen::ArrayXXd &,
                                        std::vector<Eigen::ArrayXXd> &forward_eval)
      {
        return forward_eval[param1].cosh();
      }

      void cosh_reverse_eval(int reverse_index, int param1, int,
                             const std::vector<Eigen::ArrayXXd> &forward_eval,
                             std::vector<Eigen::ArrayXXd> &reverse_eval)
      {
        Eigen::ArrayXXd fe1 = broadcast(forward_eval[param1], 
                                        reverse_eval[reverse_index].rows(),
                                        reverse_eval[reverse_index].cols());
        reverse_eval[param1] += reverse_eval[reverse_index] * fe1.sinh();
      }

    } // namespace

    Eigen::ArrayXXd ForwardEvalFunction(int node, int param1, int param2,
                                        const Eigen::ArrayXXd &x,
                                        const Eigen::ArrayXXd &constants,
                                        std::vector<Eigen::ArrayXXd> &forward_eval)
    {
      switch (node)
      {
      case Op::kInteger:
        return integer_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kVariable:
        return loadx_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kConstant:
        return loadc_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kAddition:
        return add_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kSubtraction:
        return subtract_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kMultiplication:
        return multiply_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kDivision:
        return divide_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kSin:
        return sin_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kCos:
        return cos_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kExponential:
        return exp_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kLogarithm:
        return log_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kPower:
        return pow_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kAbs:
        return abs_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kSqrt:
        return sqrt_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kSafePower:
        return safepow_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kSinh:
        return sinh_forward_eval(param1, param2, x, constants, forward_eval);
      case Op::kCosh:
        return cosh_forward_eval(param1, param2, x, constants, forward_eval);
      }
      throw std::runtime_error("Unknown Operator In Forward Evaluation");
    }

    void ReverseEvalFunction(int node, int reverse_index, int param1, int param2,
                             const std::vector<Eigen::ArrayXXd> &forward_eval,
                             std::vector<Eigen::ArrayXXd> &reverse_eval)
    {
      switch (node)
      {
      case Op::kInteger:
        return integer_reverse_eval(reverse_index, param1, param2, forward_eval,
                                    reverse_eval);
      case Op::kVariable:
        return loadx_reverse_eval(reverse_index, param1, param2, forward_eval,
                                  reverse_eval);
      case Op::kConstant:
        return loadc_reverse_eval(reverse_index, param1, param2, forward_eval,
                                  reverse_eval);
      case Op::kAddition:
        return add_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
      case Op::kSubtraction:
        return subtract_reverse_eval(reverse_index, param1, param2, forward_eval,
                                     reverse_eval);
      case Op::kMultiplication:
        return multiply_reverse_eval(reverse_index, param1, param2, forward_eval,
                                     reverse_eval);
      case Op::kDivision:
        return divide_reverse_eval(reverse_index, param1, param2, forward_eval,
                                   reverse_eval);
      case Op::kSin:
        return sin_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
      case Op::kCos:
        return cos_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
      case Op::kExponential:
        return exp_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
      case Op::kLogarithm:
        return log_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
      case Op::kPower:
        return pow_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
      case Op::kAbs:
        return abs_reverse_eval(reverse_index, param1, param2, forward_eval,
                                reverse_eval);
      case Op::kSqrt:
        return sqrt_reverse_eval(reverse_index, param1, param2, forward_eval,
                                 reverse_eval);
      case Op::kSafePower:
        return safepow_reverse_eval(reverse_index, param1, param2, forward_eval,
                                    reverse_eval);
      case Op::kSinh:
        return sinh_reverse_eval(reverse_index, param1, param2, forward_eval,
                                 reverse_eval);
      case Op::kCosh:
        return cosh_reverse_eval(reverse_index, param1, param2, forward_eval,
                                 reverse_eval);
      }
      throw std::runtime_error("Unknown Operator In Reverse Evaluation");
    }
  } // namespace backend
} // namespace bingo