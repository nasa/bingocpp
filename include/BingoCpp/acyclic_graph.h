/*!
 * \file acyclic_graph.h
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This is the header file for the acyclic graph representation of a symbolic
 * equation.
 *
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

#ifndef INCLUDE_BINGOCPP_ACYCLIC_GRAPH_H_
#define INCLUDE_BINGOCPP_ACYCLIC_GRAPH_H_

#include <set>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>

namespace bingo {

/*! \class AcyclicGraph
 *
 *  Acyclic Graph representation of an equation.
 *
 *  \note Operators include : X_Load, C_Load, Addition, Subtraction,
 *        Multiplication, Division, sin, cos, exp, log, pow, abs, sqrt
 *
 *  \fn bool needs_optimization()
 *  \fn void set_constants(Eigen::VectorXd con)
 *  \fn int count_constants()
 *  \fn Eigen::ArrayXXd evaluate(Eigen::ArrayXXd &eval_x)
 *  \fn std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> evaluate_deriv(Eigen::ArrayXXd &eval_x)
 *  \fn std::string latexstring()
 *  \fn std::set<int> utilized_commands()
 *  \fn int complexity()
 *  \fn std::string print_stack()
 */
class AcyclicGraph {
 public:
  //! Eigen::ArrayX3i stack
  /*! stack representation of equation */
  Eigen::ArrayX3i stack;
  //! Eigen::ArrayX3i simple_stack
  /*! stack simplified stack */
  Eigen::ArrayX3i simple_stack;
  //! Eigen::VectorXd constants
  /*! vector to hold constants */
  Eigen::VectorXd constants;
  //! double fitness
  /*! holds fitness for individual */
  std::vector<double> fitness;
  //! bool fit_set
  /*! if the fitness is set */
  bool fit_set;
  //! bool needs_opt
  /*! if the constants need optimization */
  bool needs_opt;
  //! int op_rate
  /*! rate to determine when to optimize
   *
   * 0 - Default - no extra optimization
   * 1 - Simplify stack inputs constants and sets needs optimization
   * 2 - Same as 1, but during crossover, bring constants from parent to child
   * 3 - Same as 1, but optimize every crossover
   * 4 - Same as 1, but optimize every mutation
   * 5 - Same as 1, but optimize every mutation and crossover
   */
  int opt_rate;
  //! int genetic_age
  /*! holds genetic age of individual */
  int genetic_age;

  
    
  //! \brief Default constructor
  AcyclicGraph();
  //! \brief Copy constructor
  AcyclicGraph(const AcyclicGraph &ag);
  //! \brief Copies self (for pybind)
  AcyclicGraph copy();
  /*! \brief find out whether constants need optimization
   *
   *  \return true if stack needs optimization
   */
  bool needs_optimization();
  /*! \brief set the constants
   *
   *  \param[in] con The constants to set. Eigen::VectorXd
   */
  void set_constants(Eigen::VectorXd con);
  /*! \brief returns constants.size()
   *
   *  \return int the size of the constants vector
   */
  int count_constants();
  /*! \brief replaces -1 in stack with location in constants vector
   *
   *  \return void
   */
  // void input_constants();
  /*! \brief evaluate the compiled stack
   *
   *  \param[in] eval_x The x parameters. Eigen::ArrayXXd
   *  \return Eigen::ArrayXXd of evaluated stack
   */
  Eigen::ArrayXXd evaluate(Eigen::ArrayXXd &eval_x);
  /*! \brief evaluate the compiled stack
   *
   *  \param[in] eval_x The x parameters. Eigen::ArrayXXd
   *  \return std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> of evaluated deriv stack
   */
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd>evaluate_with_const_deriv(
    Eigen::ArrayXXd &eval_x);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> evaluate_deriv(
    Eigen::ArrayXXd &eval_x);
  /*! \brief conversion to simplified latex string
   *
   *  \return the latexstring representation of the stack
   */
  std::string latexstring();
  /*! \brief find which commands are utilized
   *
   *  \return std::set<int> with the numbers that are utilized
   */
  std::set<int> utilized_commands();
  /*! \brief find number of commands that are utilized
   *
   *  \return int the complexity
   */
  int complexity();
  /*! \brief string output
   *
   *  \return the string to display the stack
   */
  std::string print_stack();

  static bool has_arity_two(int node);
  static bool is_terminal(int node);
  static const char *get_print(int node);

 private:
  static const bool is_arity_2_map[13]; 
  static const bool is_terminal_map[13];
  static const char *stack_print_map[13];
};

} // Namespace bingo
#endif  // INCLUDE_BINGOCPP_ACYCLIC_GRAPH_H_

