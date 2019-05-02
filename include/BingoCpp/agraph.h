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
#ifndef INCLUDE_BINGOCPP_AGRAPH_H_
#define INCLUDE_BINGOCPP_AGRAPH_H_

#include <set>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>

typedef std::unordered_map<int, std::string> PrintMap;
typedef std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvalAndDerivative;

namespace bingo {

/**
 * @brief Acyclic graph represetnation of an equation.
 * 
 * This class contains most of the code necessary for the representation of an
 * acyclic graph (linear stack) in symbolic regression.
 */
class AGraph {
 public:
  AGraph();

  AGraph(const AGraph& agraph);

  /**
   * @brief Creates a copy of this AGraph
   * 
   * @return AGraph
   */
  AGraph copy();

  /**
   * @brief Get the Command Array object
   * 
   * @return Eigen::ArrayX3i The command array for this graph.
   */
  Eigen::ArrayX3i getCommandArray() const;
  
  /**
   * @brief Set the Command Array object
   * 
   * @param command_array A copy of the new Command Array
   */
  void setCommandArray(Eigen::ArrayX3i command_array);

  /**
   * @brief Nofity individual of inplace modification of command array.
   * 
   */
  void notifyCommandArrayModificiation();

  /**
   * @brief Get the Fitness of this AGraph
   * 
   * @return double 
   */
  double getFitness() const;

  /**
   * @brief Set the Fitness for this AGraph
   * 
   * @param fitness 
   */
  void setFitness(double fitness);

  /**
   * @brief Check if fitness has been set for this AGraph object.
   * 
   * @return true if set
   * @return false otherwise
   */
  bool isFitnessSet() const;

  /**
   * @brief Set the Genetic Age of this AGraph
   * 
   * @param age The age of this AGraph
   */
  void setGeneticAge(const int age);

  /**
   * @brief Get the Genetic Age of this AGraph
   * 
   * @return int 
   */
  int getGeneticAge() const;

  /**
   * @brief Get the Utilized Commands for the CommandArray
   * 
   * Returns a mask of all the utilized commands in the stack for 
   * representing the AGraph. The indicies in the vector represent the
   * indicies in the CommandArray
   * 
   * @return std::vector<bool> The mask.
   */
  std::vector<bool> getUtilizedCommands() const;

  /**
   * @brief The AGraph needs local optimization.
   * 
   * Determine if the agraph needs local optimzation.
   * 
   * @return true Needs optimization.
   * @return false Has been optimized
   */
  bool needsLocalOptimization() const;

  /**
   * @brief Get the Number Local Optimization Params
   * 
   * The number of parameters that need to be optimized in this 
   * AGraph.
   * 
   * @return int The number of parameters to optimize.
   */
  int getNumberLocalOptimizationParams() const;

  /**
   * @brief Set the Local Optimization Params
   * 
   * Set the optimized constants
   * 
   * @param params The optimized constants.
   */
  void setLocalOptimizationParams(Eigen::VectorXd params);

  /**
   * @brief Get the constants in the graph.
   * 
   * Returns the constants in the graph. The AGraph should be optimized
   * before calling this method.
   * 
   * @return Eigen::VectorXd The constants in the AGraph
   */
  Eigen::VectorXd getLocalOptimizationParams() const;

  /**
   * @brief Evaluate the AGraph equatoin
   * 
   * Evaluation of the AGraph Equation at points x.
   * 
   * @param x Values at which to evaluate the equations. x is MxD where D is the 
   * number of dimensions in x and M is the number of data points in x.
   * 
   * @return Eigen::ArrayXXd The evaluation of function at points x.
   */
  Eigen::ArrayXXd evaluateEquationAt(Eigen::ArrayXXd& x);

  /**
   * @brief Evaluate the AGraph and get its derivatives
   * 
   * Evaluation of the AGraph equation along points x and the graident
   * of the equation with respect to x.
   * 
   * @param x Values at which to evaluate the equations. x is MxD where D is the 
   * number of dimensions in x and M is the number of data points in x.
   * 
   * @return EvalAndDerivative The evaluation of the function of this AGraph
   * along the points x and the derivative of the equation with respect to x.
   */
  EvalAndDerivative evaluateEquationWithXGradientAt(Eigen::ArrayXXd& x);

  /**
   * @brief Evluate the AGraph and get its derivatives.
   * 
   * Evaluation of the AGraph equation along the points x and the gradient
   * of the equation with respect to the constants of the equation.
   * 
   * @param x Values at which to evaluate the equations. x is MxD where D is the 
   * number of dimensions in x and M is the number of data points in x.
   * 
   * @return EvalAndDerivative The evaluation of the function of this AGraph
   * along the points x and the derivative of the equation with respect to 
   * the constants of the equation.
   */
  EvalAndDerivative evaluateEquationWithLocalOptGradientAt(Eigen::ArrayXXd& x);

  /**
   * @brief Get the Latex String of this AGraph equation.
   * 
   * @return std::string 
   */
  std::string getLatexString() const;

  /**
   * @brief Get the Console String this AGraph equation.
   * 
   * @return std::string 
   */
  std::string getConsoleString() const;

  /**
   * @brief Get the Stack String this AGraph equation.
   * 
   * @return std::string 
   */
  std::string getStackString() const;

  /**
   * @brief Get the Complexity of this AGraph equation.
   * 
   * @return int 
   */
  int getComplexity() const;

  /**
   * @brief Determines if the equation operation has arity two.
   * 
   * @param node The operation of the equation.
   * @return true If the operation requires to parameters.
   * @return false Otherwise.
   */
  static bool hasArityTwo(int node);

  /**
   * @brief Determines if the equation operation is loading a value.
   * 
   * @param node The operation of the equation.
   * @return true If the node loads a value.
   * @return false It has arity greater than 0.
   */
  static bool isTerminal(int node);

 private:
  Eigen::ArrayX3i command_array_;
  Eigen::ArrayX3i short_command_array_;
  Eigen::VectorXd constants_;
  bool needs_opt_;
  int num_constants_;
  double fitness_;
  bool fit_set_;
  int genetic_age_;

  // To string operator when passed into stream
  friend std::ostream& operator<<(std::ostream&, const AGraph&);

  // Maps for operation nodes.
  static const bool kIsArity2Map[13]; 
  static const bool kIsTerminalMap[13];

  // Helper Functions
  void process_modified_command_array();
  void renumber_constants(const std::vector<bool>& utilized_commands);
  void update_short_command_array(const std::vector<bool>& utilized_commands);
  std::string get_stack_string(const bool is_short=false) const;
  std::string get_formatted_string_using(const PrintMap& format_map) const;
};
} // namespace bingo
#endif //INCLUDE_BINGOCPP_AGRAPH_H_
