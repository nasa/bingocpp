/*!
 * \file acyclic_graph.hh
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This is the header file for the acyclic graph representation of a symbolic
 * equation.
 */

#ifndef INCLUDE_BINGOCPP_ACYCLIC_GRAPH_H_
#define INCLUDE_BINGOCPP_ACYCLIC_GRAPH_H_

#include <Eigen/Dense>
#include <Eigen/Core>

#include <set>
#include <utility>
#include <vector>



typedef Eigen::ArrayX3d CommandStack;
typedef Eigen::Ref<Eigen::ArrayXXd,
        0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> ArrayByRef;


// using EigenDStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
// template <typename MatrixType> using EigenDRef =
//    Eigen::Ref<MatrixType, 0, EigenDStride>;


/*!
 * \brief Computes reverse autodiff partial of a stack command.
 *
 * The partial derivative of the result with respect to the command at the
 * specified location in the stack is evaluated.  This requires the addition of
 * all the dependencies of the command using the chain rule.  References can be
 * made in to the forward buffer and later reverse buffer values; all are
 * referenced by index.
 *
 * \note Addition of new operators must edit this segment.
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 * \param[in] command_index Index of command in the stack; also the location of
 *                          the result to be placed in the reverse buffer.
 * \param[in] forward_buffer Vector of Eigen arrays for the forward buffer.
 * \param[in\out] forward_buffer Vector of Eigen arrays for the forward buffer.
 * \param[in] dependencies Vector of indices of the stack which depend on the
                           specified command.
 *
 */
void ReverseSingleCommand(const CommandStack &stack,
                          const int command_index,
                          const std::vector<Eigen::ArrayXXd> &forward_buffer,
                          std::vector<Eigen::ArrayXXd> &reverse_buffer,
                          const std::set<int> &dependencies);


/*!
 * \brief Evaluates a stack at the given x using the given constants.
 *
 * An acyclic graph is given in stack form.  The stack is evaluated command by
 * command putting the result of each command into a local buffer.  References
 * can be made in the stack to columns of the x input as well as constants; both
 * are referenced by index.
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 * \param[in] x The input variables to the acyclic graph. (Eigen::ArrayXXd)
 * \param[in] constants Vector of the constants used in the stack.
 *
 * \return The value of the last command in the stack. (Eigen::ArrayXXd)
 */
Eigen::ArrayXXd Evaluate(const CommandStack &stack,
                         const Eigen::ArrayXXd &x,
                         const std::vector<double> &constants);


/*!
 * \brief Evaluates a stack, but only the commands that are utilized.
 *
 * An acyclic graph is given in stack form.  The stack is evaluated, but only
 * the commands which are utilized by the final result.
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 * \param[in] x The input variables to the acyclic graph. (Eigen::ArrayXXd)
 * \param[in] constants Vector of the constants used in the stack.
 *
 * \return The value of the last command in the stack. (Eigen::ArrayXXd)
 */
Eigen::ArrayXXd SimplifyAndEvaluate(const CommandStack &stack,
                                    const Eigen::ArrayXXd &x,
                                    const std::vector<double> &constants);


/*!
 * \brief Evaluates a stack at the given x using the given constants.
 *
 * An acyclic graph is given in stack form.  The stack is evaluated command by
 * command (but only the ones with a true value of the mask) putting the result
 * of each command into a local buffer.  References can be made in the satck to
 * columns of the x input as well as constants; both are referenced by index.
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 * \param[in] x The input variables to the acyclic graph. (Eigen::ArrayXXd)
 * \param[in] constants Vector of the constants used in the stack.
 * \param[in] mask Vector of booleans detailing which commands are included.
 *
 * \return The value of the last command in the stack. (Eigen::ArrayXXd)
 */
Eigen::ArrayXXd EvaluateWithMask(const CommandStack &stack,
                                 const Eigen::ArrayXXd &x,
                                 const std::vector<double> &constants,
                                 const std::vector<bool> &mask);



/*!
 * \brief Evaluates a stack and its derivative with the given x and constants.
 *
 * An acyclic graph is given in stack form.  The stack is evaluated command by
 * command putting the result of each command into a local buffer.  References
 * can be made in the satck to columns of the x input as well as constants; both
 * are referenced by index.  The stack is then processed in reverse to calculate
 * the gradient of the stack with respect to x.  This reverse processing is
 * standard reverse auto-differentiation.
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 * \param[in] x The input variables to the acyclic graph. (Eigen::ArrayXXd)
 * \param[in] constants Vector of the constants used in the stack.
 *
 * \return The value of the last command in the stack and the gradient.
 *         (std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd>)
 */
std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvaluateWithDerivative(
  const CommandStack &stack,
  const Eigen::ArrayXXd &x,
  const std::vector<double> &constants);



/*!
 * \brief Evaluates a stack and its derivative, but only the utilized commands.
 *
 * An acyclic graph is given in stack form.  The stack is evaluated with its
 * derivative, but only the commands which are utilized by the final result.
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 * \param[in] x The input variables to the acyclic graph. (Eigen::ArrayXXd)
 * \param[in] constants Vector of the constants used in the stack.
 *
 * \return The value of the last command in the stack and the gradient.
 *         (std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd>)
 */
std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> SimplifyAndEvaluateWithDerivative(
  const CommandStack &stack,
  const Eigen::ArrayXXd &x,
  const std::vector<double> &constants);



/*!
 * \brief Evaluates a stack and its derivative with the given x and constants.
 *
 * An acyclic graph is given in stack form.  The stack is evaluated command by
 * command (but only the ones with a true value of the mask) putting the result
 * of each command into a local buffer.  References can be made in the satck to
 * columns of the x input as well as constants; both are referenced by index.
 * The stack is then processed in reverse (again considering only the ones with
 * a true value of the mask) to calculate the gradient of the stack with respect
 * to x.  This reverse processing is standard reverse auto-differentiation.
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 * \param[in] x The input variables to the acyclic graph. (Eigen::ArrayXXd)
 * \param[in] constants Vector of the constants used in the stack.
 * \param[in] mask Vector of booleans detailing which commands are included.
 *
 * \return The value of the last command in the stack and the gradient.
 *         (std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd>)
 */
std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvaluateWithDerivativeAndMask(
  const CommandStack &stack,
  const Eigen::ArrayXXd &x,
  const std::vector<double> &constants,
  const std::vector<bool> &mask);



/*!
 * \brief Prints a stack to std::cout.
 *
 * An acyclic graph is given in stack form.  The stack is printed to std::cout
 * command by command in the following format:
 * (stack_location) = operation : (parameters)
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 */
void PrintStack(const CommandStack &stack);



/*!
 * \brief Simplifies a stack.
 *
 * An acyclic graph is given in stack form.  The stack is first simplified to
 * consist only of the commands used by the last command.
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 *
 * \return Simplified stack.
 */
CommandStack SimplifyStack(const CommandStack &stack);



/*!
 * \brief Finds which commands are utilized in a stack.
 *
 * An acyclic graph is given in stack form.  The stack is processed in reverse
 * to find which commands the last command depends.
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 *
 * \return Vector describing which commands in the stack are used.
 */
std::vector<bool> FindUsedCommands(const CommandStack &stack);

#endif  // INCLUDE_BINGOCPP_ACYCLIC_GRAPH_H_

