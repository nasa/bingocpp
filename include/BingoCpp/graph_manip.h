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

#include "BingoCpp/backend.h"
#include "BingoCpp/acyclic_graph.h"

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
   * 2 - Same as 1, but during crossover, bring constants from parent to child
   * 3 - Same as 1, but optimize every crossover
   * 4 - Same as 1, but optimize every mutation
   * 5 - Same as 1, but optimize every mutation and crossover
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
                          int opt_rate = 0);
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
  // std::pair<Eigen::ArrayX3i, Eigen::VectorXd> dump(AcyclicGraph &indv);
  std::pair<std::pair<Eigen::ArrayX3i, Eigen::VectorXd>, int> dump(
    AcyclicGraph &indv);
  /*! \brief loads a new AcyclicGraph with the stack and constants
   *
   *  \returns AcyclicGraph with the stack and constants
   */
  AcyclicGraph load(std::pair<std::pair<Eigen::ArrayX3i, Eigen::VectorXd>, int>
                    indv_list);
  // AcyclicGraph load(std::pair<Eigen::ArrayX3i, Eigen::VectorXd> indv_list);
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
