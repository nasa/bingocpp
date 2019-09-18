#ifndef BINGOCPP_INCLUDE_BINGOCPP_EQUATION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_EQUATION_H_

#include <string>

#include <Eigen/Dense>

namespace bingo {
 
typedef std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvalAndDerivative;

class Equation {
 public:
    /**
   * @brief Evaluate the Equation
   * 
   * Evaluation of the Equation at points x.
   * 
   * @param x Values at which to evaluate the equations. x is MxD where D is the 
   * number of dimensions in x and M is the number of data points in x.
   * 
   * @return Eigen::ArrayXXd The evaluation of function at points x.
   */
  virtual Eigen::ArrayXXd 
  EvaluateEquationAt(const Eigen::ArrayXXd& x) const = 0;

  /**
   * @brief Evaluate the Equation and get its derivatives
   * 
   * Evaluation of the Equation along points x and the graident
   * of the equation with respect to x.
   * 
   * @param x Values at which to evaluate the equations. x is MxD where D is the 
   * number of dimensions in x and M is the number of data points in x.
   * 
   * @return EvalAndDerivative The evaluation of the function of this Equation 
   * along the points x and the derivative of the equation with respect to x.
   */
  virtual EvalAndDerivative
  EvaluateEquationWithXGradientAt(const Eigen::ArrayXXd& x) const = 0;

  /**
   * @brief Evaluate the Equation and get its derivatives.
   * 
   * Evaluation of the this Equation along the points x and the gradient
   * of the equation with respect to the constants of the equation.
   * 
   * @param x Values at which to evaluate the equations. x is MxD where D is the 
   * number of dimensions in x and M is the number of data points in x.
   * 
   * @return EvalAndDerivative The evaluation of the function of this Equation 
   * along the points x and the derivative of the equation with respect to 
   * the constants of the equation.
   */
  virtual EvalAndDerivative
  EvaluateEquationWithLocalOptGradientAt(const Eigen::ArrayXXd& x) const = 0;

  /**
   * @brief Get the Latex String of this Equation.
   * 
   * @return std::string 
   */
  virtual std::string GetLatexString() const = 0;

  /**
   * @brief Get the Console String this Equation.
   * 
   * @return std::string 
   */
  virtual std::string GetConsoleString() const = 0;

  /**
   * @brief Get the Stack String this Equation.
   * 
   * @return std::string 
   */
  virtual std::string GetStackString() const = 0;

  /**
   * @brief Get the Complexity of this Equation.
   * 
   * @return int 
   */
  virtual int GetComplexity() const = 0;

};
} // namespace bingo
#endif //BINGOCPP_INCLUDE_BINGOCPP_EQUATION_H_