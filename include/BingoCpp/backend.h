/*
 * Copyright 2018 United States Government as represented by the Administrator 
 * of the National Aeronautics and Space Administration. No copyright is claimed 
 * in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 *
 * The Bingo Mini-app platform is licensed under the Apache License, Version 2.0 
 * (the "License"); you may not use this file except in compliance with the 
 * License. You may obtain a copy of the License at  
 * http://www.apache.org/licenses/LICENSE-2.0. 
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
 * License for the specific language governing permissions and limitations under 
 * the License.
*/
#ifndef INCLUDE_BINGOCPP_BACKEND_H_
#define INCLUDE_BINGOCPP_BACKEND_H_

#include <set>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <BingoCpp/agraph.h>

namespace bingo {
namespace backend{
/*!
 * \brief Identify whether a c++ backend is being used in python module.
 *
 * \return true, the backend is c++ (bool)
 */
bool isCpp();

/*!
 * \brief Evaluates a stack at the given x using the given constants.
 *
 * An acyclic graph is given in stack form.  The stack is evaluated command by
 * command putting the result of each command into a local buffer.  References
 * can be made in the stack to columns of the x input as well as constants; both
 * are referenced by index.
 *
 * \param stack Description of an acyclic graph in stack format.
 * \param x The input variables to the acyclic graph. (Eigen::ArrayXXd)
 * \param constants Vector of the constants used in the stack.
 *
 * \return The value of the last command in the stack. (Eigen::ArrayXXd)
 */
Eigen::ArrayXXd evaluate(const Eigen::ArrayX3i& stack,
                         const Eigen::ArrayXXd& x,
                         const Eigen::VectorXd& constants);

/*!
 * \brief Evaluates a stack and its derivative with the given x and constants.
 *
 * An acyclic graph is given in stack form.  The stack is evaluated command by
 * command putting the result of each command into a local buffer.  References
 * can be made in the stack to columns of the x input as well as constants; both
 * are referenced by index.  The stack is then processed in reverse to calculate
 * the gradient of the stack with respect to the chosen parameter.  This reverse 
 * processing is standard reverse auto-differentiation.
 *
 * \param stack Description of an acyclic graph in stack format.
 * \param x The input variables to the acyclic graph. (Eigen::ArrayXXd)
 * \param constants Vector of the constants used in the stack.
 * \param param_x_or_c true: x derivative, false: c derivative
 *
 * \return The value of the last command in the stack and the gradient.
 *         (std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd>)
 */
EvalAndDerivative evaluateWithDerivative(
    const Eigen::ArrayX3i& stack,
    const Eigen::ArrayXXd& x,
    const Eigen::VectorXd& constants,
    const bool param_x_or_c = true);


/*!
 * \brief Evaluates a stack, but only the commands that are utilized.
 *
 * An acyclic graph is given in stack form.  The stack is evaluated, but only
 * the commands which are utilized by the final result.
 *
 * \param stack Description of an acyclic graph in stack format.
 * \param x The input variables to the acyclic graph. (Eigen::ArrayXXd)
 * \param constants Vector of the constants used in the stack.
 *
 * \return The value of the last command in the stack. (Eigen::ArrayXXd)
 */
Eigen::ArrayXXd simplifyAndEvaluate(const Eigen::ArrayX3i& stack,
                                    const Eigen::ArrayXXd& x,
                                    const Eigen::VectorXd& constants);


/*!
 * \brief Evaluates a stack and its derivative, but only the utilized commands.
 *
 * An acyclic graph is given in stack form.  The stack is evaluated with its
 * derivative, but only the commands which are utilized by the final result.
 *
 * \param stack Description of an acyclic graph in stack format.
 * \param x The input variables to the acyclic graph. (Eigen::ArrayXXd)
 * \param constants Vector of the constants used in the stack.
 * \param param_x_or_c true: x derivative, false: c derivative
 *
 * \return The value of the last command in the stack and the gradient.
 */
EvalAndDerivative simplifyAndEvaluateWithDerivative(
    const Eigen::ArrayX3i& stack,
    const Eigen::ArrayXXd& x,
    const Eigen::VectorXd& constants,
    const bool param_x_or_c = true);


/*!
 * \brief Simplifies a stack.
 *
 * An acyclic graph is given in stack form.  The stack is first simplified to
 * consist only of the commands used by the last command.
 *
 * \param stack Description of an acyclic graph in stack format.
 *
 * \return Simplified stack.
 */
Eigen::ArrayX3i simplifyStack(const Eigen::ArrayX3i& stack);


/*!
 * \brief Finds which commands are utilized in a stack.
 *
 * An acyclic graph is given in stack form.  The stack is processed in reverse
 * to find which commands the last command depends.
 *
 * \param stack Description of an acyclic graph in stack format.
 *
 * \return vector describing which commands in the stack are used.
 */
std::vector<bool> getUtilizedCommands(const Eigen::ArrayX3i& stack);
} // namespace backend
} // namespace bingo
#endif  
