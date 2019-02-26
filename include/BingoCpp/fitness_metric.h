/*!
 * \file fitness_metric.h
 *
 * \author Ethan Adams
 * \date
 *
 * This file contains the cpp version of FitnessMetric.py
 */

#ifndef INCLUDE_BINGOCPP_FITNESS_METRIC_H_
#define INCLUDE_BINGOCPP_FITNESS_METRIC_H_

#include "BingoCpp/acyclic_graph.h"
#include "BingoCpp/training_data.h"
#include <Eigen/Dense>
#include <Eigen/Core>

struct FitnessMetric;

/*! \struct LMFunctor
 *
 *  Used for Levenberg-Marquardt Optimization
 *
 *  \fn int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec)
 *  \fn int df(const Eigen::VectorXd &x, Eigen::MatrixXf &fjac)
 *  \fn int values() const
 *  \fn int inputs() const
 */
struct LMFunctor {
  //! int m
  /*! Number of data points, i.e. values */
  int m;
  //! int n
  /*! Number of parameters, i.e. inputs */
  int n;
  //! AcyclicGraph indv
  /*! The Agraph individual */
  AcyclicGraph agraphIndv;
  //! TrainingData* train
  /*! object that holds data needed */
  TrainingData* train;
  //! FitnessMetric* fit
  /*! object that holds fitness metric */
  FitnessMetric* fit;
  /*! \brief Compute 'm' errors, one for each data point, for the given paramter values in 'x'
   *
   *  \param[in] x contains current estimates for parameters. Eigen::VectorXd (dimensions nx1)
   *  \param[in] fvec contain error for each data point. Eigen::VectorXd (dimensions mx1)
   *  \return 0
   */
  int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec);
  /*! \brief Compute jacobian of the errors
   *
   *  \param[in] x contains current estimates for parameters. Eigen::VectorXd (dimensions nx1)
   *  \param[in] fjac contain jacobian of the errors, calculated numerically. Eigen::MatrixXf (dimensions mxn)
   *  \return 0
   */
  int df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac);
  /*! \brief gets the values
   *
   *  \return m - values
   */
  int values() const {
    return m;
  }
  /*! \brief gets the inputs
   *
   *  \return n - inputs
   */
  int inputs() const {
    return n;
  }

};

/*! \struct FitnessMetric
 *
 *  An abstract struct to evaluate metric based on type of regression
 *
 *  \note FitnessMetric includes : StandardRegression
 *
 *  \fn virtual Eigen::ArrayXXd evaluate_fitness_vector(AcyclicGraph &indv, TrainingData &train) = 0
 *  \fn float evaluate_fitness(AcyclicGraph &indv, TrainingData &train)
 *  \fn void optimize_constants(AcyclicGraph &indv, TrainingData &train)
 */
struct FitnessMetric {
 public:
  FitnessMetric() { }
  /*! \brief f(x) - y where f is defined by indv and x, y are in train
  *
  *  \note Each implementation will need to hard code casting TrainingData
  *        to a specific type in this function.
  *
  *  \param[in] indv agcpp indv to be evaluated. AcyclicGraph
  *  \param[in] train The TrainingData to evaluate the fitness. TrainingData
  *  \return Eigen::ArrayXXd the fitness vector
  */
  virtual Eigen::ArrayXXd evaluate_fitness_vector(AcyclicGraph &indv,
      TrainingData &train) = 0;
  /*! \brief Finds the fitness metric
  *
  *  \param[in] indv agcpp indv to be evaluated. AcyclicGraph
  *  \param[in] train The TrainingData to evaluate the fitness. TrainingData
  *  \return float the fitness metric
  */
  double evaluate_fitness(AcyclicGraph &indv, TrainingData &train);
  /*! \brief perform levenberg-marquardt optimization on embedded constants
  *
  *  \param[in] indv agcpp indv to be evaluated. AcyclicGraph
  *  \param[in] train The TrainingData used by fitness metric. TrainingData
  */
  void optimize_constants(AcyclicGraph &indv, TrainingData &train);
};

/*! \struct StandardRegression
 *  \brief Traditional fitness evaluation
 */
struct StandardRegression : FitnessMetric {
 public:
  StandardRegression() : FitnessMetric() {}
  Eigen::ArrayXXd evaluate_fitness_vector(AcyclicGraph &indv,
                                          TrainingData &train);
};

/*! \struct ImplicitRegression
 *  \brief Implicit Regression
 */
struct ImplicitRegression : FitnessMetric {
 public:
  //! int required_params
  /*! minimum number of non zero components of dot */
  int required_params;
  //! bool normalize_dot
  /*! normalize the terms in the dot product */
  bool normalize_dot;
  //! double acceptable_finite_fracion
  /*! yea */
  double acceptable_finite_fracion;
  // ImplicitRegression() : FitnessMetric() {}
  ImplicitRegression(int required_params = 0, bool normalize_dot = false,
                     double acceptable_nans = 0.1);
  Eigen::ArrayXXd evaluate_fitness_vector(AcyclicGraph &indv,
                                          TrainingData &train);
};

#endif