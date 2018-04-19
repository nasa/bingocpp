/*!
 * \file graph_manip.h
 *
 * \author Ethan Adams
 * \date 2/26/2018
 *
 * This file contains the cpp version of AGraphCpp.py
 */

#ifndef INCLUDE_BINGOCPP_GRAPH_MANIP_H_
#define INCLUDE_BINGOCPP_GRAPH_MANIP_H_

#include <Eigen/Dense>
#include <Eigen/Core>

#include <set>
#include <map>
#include <utility>
#include <vector>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <iomanip>
#include <time.h>

#include "BingoCpp/acyclic_graph_nodes.h"



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
  //! Eigen::ArrayX3d stack
  /*! stack representation of equation */
  Eigen::ArrayX3d stack;
  //! Eigen::ArrayX3d simple_stack
  /*! stack simplified stack */
  Eigen::ArrayX3d simple_stack;
  //! Eigen::VectorXd constants
  /*! vector to hold constants */
  Eigen::VectorXd constants;
  //! OperatorInterface oper_interface
  /*! map that holds operators with arity / strings */
  OperatorInterface oper_interface;
  //! double fitness
  /*! holds fitness for individual */
  std::vector<double> fitness;
  //! bool fit_set
  /*! if the fitness is set */
  bool fit_set;
  //! bool needs_opt
  /*! if the constants need optimization */
  bool needs_opt;
  //! int opt_rate
  /*! holds rate of optimization */
  int opt_rate;
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
};

/*! \class AcyclicGraphManipulator
 *
 *  Manipulates AGraph objects for generation, crossover, mutation, and distance
 *
 *  \fn void add_node_type(int node_type)
 *  \fn AcyclicGraph generate()
 *  \fn std::vector<AcyclicGraph> crossover(AcyclicGraph &parent1, AcyclicGraph &parent2)
 *  \fn void mutation(AcyclicGraph &indv)
 *  \fn int distance(AcyclicGraph &indv1, AcyclicGraph &indv2)
 *  \fn std::vector<int> rand_operator_params(int arity, int stack_location)
 *  \fn int rand_operator_type()
 *  \fn std::vector<int> rand_operator(int stack_location)
 *  \fn int rand_terminal_param(int terminal)
 *  \fn int mutate_terminal_param(int terminal)
 *  \fn std::vector<int> rand_terminal()
 */
class AcyclicGraphManipulator {
 public:
  //! int nvars
  /*! number of variables */
  int nvars;
  //! int ag_size
  /*! the size of the AGraph */
  int ag_size;
  //! int nloads
  /*! number of loads */
  int nloads;
  //! float float_lim
  /*! float limit */
  float float_lim;
  //! float terminal_prob
  /*! float to hold probability */
  float terminal_prob;
  //! int op_rate
  /*! optimization rate to give to generated AGraphs 
   *
   * 0 - Default - no extra optimization 
   * 1 - Simplify stack inputs constants and sets needs optimization
   * 
   */
  int opt_rate;
  //! std::vector<int> node_type_vec
  /*! vector to hold the types of nodes in the manipulator */
  std::vector<int> node_type_vec;
  //! std::vector<int> op_vec
  /*! vector to hold the types of operators used in the manipulator */
  std::vector<int> op_vec;
  //! std::vector<int> term_vec
  /*! vector to hold the types of terminals used in the manipulator */
  std::vector<int> term_vec;
  //! int num_node_types
  /*! int to hold the number of node types (matches the python) */
  int num_node_types;

  //! \brief Constructor
  AcyclicGraphManipulator(int nvars = 3, int ag_size = 15, int nloads = 1,
                          float float_lim = 10.0, float terminal_prob = 0.1,
                          int opt_rate = 1);
  /*! \brief Add a type of node to the set of allowed types
   *
   *  \param[in] node_type The type of node to add. int
   */
  void add_node_type(int node_type);
  /*! \brief Generates random individual. Fills stack based on random
   *         nodes/terminals and random parameters
   *
   *  \returns new AcyclicGraph individual
   */
  AcyclicGraph generate();  
  /*! \brief simplifies the individual's stack.
   *
   *  \param[in] indv The individual with the stack to simplify. AcyclicGraph
   */
  void simplify_stack(AcyclicGraph &indv);
  /*! \brief takes the stack and constants into one object
   *
   *  \returns pair with the stack and constants
   */
  std::pair<Eigen::ArrayX3d, Eigen::VectorXd> dump(AcyclicGraph &indv);
  /*! \brief loads a new AcyclicGraph with the stack and constants
   *
   *  \returns AcyclicGraph with the stack and constants
   */
  AcyclicGraph load(std::pair<Eigen::ArrayX3d, Eigen::VectorXd> indv_list);
  /*! \brief Single point crossover
   *
   *  \param[in] parent1 the first parent. AcyclicGraph
   *  \param[in] parent2 the second parent. AcyclicGraph
   *  \return vector with two AcyclicGraph children (new copies)
   */
  std::vector<AcyclicGraph> crossover(AcyclicGraph &parent1,
                                      AcyclicGraph &parent2);
  /*! \brief performs 1pt mutation, does not create copy of individual
   *
   *  \param[in] indv The individual to be mutated. AcyclicGraph
   */
  AcyclicGraph mutation(AcyclicGraph &indv);
  /*! \brief Computes the distance (a measure of similarity) between two individuals
   *
   *  \param[in] indv1 first individual. AcyclicGraph
   *  \param[in] indv2 second individual. AcyclicGraph
   *  \return int the distance
   */
  int distance(AcyclicGraph &indv1, AcyclicGraph &indv2);
  /*! \brief Produces random ints for use as operator parameters
   *
   *  \param[in] arity the number of parameters needed. int
   *  \param[in] stack_location the location of command in stack. int
   *  \return vector<int> with the parameters
   */
  std::vector<int> rand_operator_params(int arity, int stack_location);
  /*! \brief Picks a random operator from the operator list
   *
   *  \return int the operator (AGraph node type)
   */
  int rand_operator_type();
  /*! \brief Produces random operator and parameters. Chooses operator from list of
   *   allowable node types
   *
   *  \param[in] stack_location the location of command in stack. int
   *  \return vector<int> with the random operator and parameters
   */
  std::vector<int> rand_operator(int stack_location);
  /*! \brief Produces random terminal value, either input variable or float
   *
   *  \param[in] terminal the terminal that needs parameters. int
   *  \return int the terminal parameters
   */
  int rand_terminal_param(int terminal);
  /*! \brief Produces random terminal value, either input variable or float
   *         Mutates floats by getting random variation of old param
   *
   *  \param[in] terminal the terminal that needs parameters mutated. int
   *  \return int the terminal parameters
   */
  int mutate_terminal_param(int terminal);
  /*! \brief Produces random terminal node and value
   *
   *  \return vector<int> with node and parameters
   */
  std::vector<int> rand_terminal();


};

//! \brief Initializes random seed (for use in python with pybind)
static void rand_init() {
  srand (time(NULL));
}



#endif