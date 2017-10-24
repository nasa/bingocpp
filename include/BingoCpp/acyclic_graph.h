/*!
 * \file acyclic_graph.hh
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This is the header file for the acyclic graph representation of a symbolic
 * equation.
 */

#ifndef ACYCLIC_GRAPH_HEADER
#define ACYCLIC_GRAPH_HEADER

#include <Eigen/Dense>


typedef std::vector< std::pair<int, std::vector<int> > > CommandStack;



/*!
 * \brief Evaluates a stack at the given x using the given constants.
 *
 * An acyclic graph is given in stack form.  The stack is evaluated command by
 * command putting the result of each command into a local buffer.  References
 * can be made in the satck to columns of the x input as well as constants; both
 * are referenced by index.
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 * \param[in] x The input variables to the acyclic graph. (Eigen::ArrayXXd)
 * \param[in] constants Vector of the constants used in the stack.
 *
 * \return The value of the last command in the stack. (Eigen::ArrayXXd)
 */
Eigen::ArrayXXd Evaluate(CommandStack stack, Eigen::ArrayXXd x,
                         std::vector<double> constants) ;


/*!
 * \brief Simplifies a stack then evaluates it.
 *
 * An acyclic graph is given in stack form.  The stack is first simplified to
 * consist only of the commands used by the last command. The stack is then
 * evaluated by calling Evaluate.
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 * \param[in] x The input variables to the acyclic graph. (Eigen::ArrayXXd)
 * \param[in] constants Vector of the constants used in the stack.
 *
 * \return The value of the last command in the stack. (Eigen::ArrayXXd)
 */
Eigen::ArrayXXd SimplifyAndEvaluate(CommandStack stack, Eigen::ArrayXXd x,
                                    std::vector<double> constants) ;



/*!
 * \brief Evaluates a stack and its derivative with the given x and constants.
 *
 * An acyclic graph is given in stack form.  The stack is evaluated command by
 * command putting the result of each command into a local buffer.  References
 * can be made in the satck to columns of the x input as well as constants; both
 * are referenced by index.  The stack is then processed in reverse to calculate
 * the gradient of the stack with  respect top x.  This reverse processing is
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
  CommandStack stack, Eigen::ArrayXXd x, std::vector<double> constants) ;



/*!
 * \brief Evaluates a stack and its derivative after simplification.
 *
 * An acyclic graph is given in stack form.  The stack is first simplified to
 * consist only of the commands used by the last command. The stack is then
 * evaluated with its derivative by calling EvaluateWithDerivative.
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 * \param[in] x The input variables to the acyclic graph. (Eigen::ArrayXXd)
 * \param[in] constants Vector of the constants used in the stack.
 *
 * \return The value of the last command in the stack and the gradient.
 *         (std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd>)
 */
std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> SimplifyAndEvaluateWithDerivative(
  CommandStack stack, Eigen::ArrayXXd x, std::vector<double> constants) ;



/*!
 * \brief Prints a stack to std::cout.
 *
 * An acyclic graph is given in stack form.  The stack is printed to std::cout
 * command by command in the following format:
 * (stack_location) = operation : (parameters)
 *
 * \param[in] stack Description of an acyclic graph in stack format.
 */
void PrintStack(CommandStack stack) ;



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
CommandStack SimplifyStack(CommandStack stack) ;



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
std::vector<bool> FindUsedCommands(CommandStack stack) ;

#endif

