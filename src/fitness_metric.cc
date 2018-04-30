/*!
 * \file fitness_metric.cc
 *
 * \author Ethan Adams
 * \date
 *
 * This file contains the cpp version of FitnessMetric.py
 */

#include "BingoCpp/fitness_metric.h"
#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/NonLinearOptimization>

int LMFunctor::operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) {
  agraphIndv.set_constants(x);
  fvec = fit->evaluate_fitness_vector(agraphIndv, *train);
  return 0;
}

int LMFunctor::df(const Eigen::VectorXd &x, Eigen::MatrixXd &fjac) {
  double epsilon;
  epsilon = 1e-5f;

  for (int i = 0; i < x.size(); i++) {
    Eigen::VectorXd xPlus(x);
    xPlus(i) += epsilon;
    Eigen::VectorXd xMinus(x);
    xMinus(i) -= epsilon;
    Eigen::VectorXd fvecPlus(values());
    operator()(xPlus, fvecPlus);
    Eigen::VectorXd fvecMinus(values());
    operator()(xMinus, fvecMinus);
    Eigen::VectorXd fvecDiff(values());
    fvecDiff = (fvecPlus - fvecMinus) / (2.0 * epsilon);
    fjac.block(0, i, values(), 1) = fvecDiff;
  }

  return 0;
}

double FitnessMetric::evaluate_fitness(AcyclicGraph &indv,
                                       TrainingData &train) {
  if (indv.needs_optimization()) {
    optimize_constants(indv, train);
  }

  return ((evaluate_fitness_vector(indv, train)).abs()).mean();
}

void FitnessMetric::optimize_constants(AcyclicGraph &indv,
                                       TrainingData &train) {
  LMFunctor functor;
  functor.train = &train;
  functor.fit = this;
  functor.m = functor.train->size();
  functor.n = indv.count_constants();
  functor.agraphIndv = indv;
  Eigen::VectorXd vec = Eigen::VectorXd::Random(functor.n);
  Eigen::LevenbergMarquardt<LMFunctor, double> lm(functor);
  lm.minimize(vec);
  indv.set_constants(vec);
  indv.needs_opt = false;
}

Eigen::ArrayXXd StandardRegression::evaluate_fitness_vector(AcyclicGraph &indv,
    TrainingData &train) {
  ExplicitTrainingData* temp = dynamic_cast<ExplicitTrainingData*>(&train);
  return (indv.evaluate(temp->x)) - temp->y;
}

ImplicitRegression::ImplicitRegression(int required_params, bool normalize_dot,
                                       double acceptable_nans) {
  this->required_params = required_params;
  this->normalize_dot = normalize_dot;
  acceptable_finite_fracion = 1.0 - acceptable_nans;
}

Eigen::ArrayXXd ImplicitRegression::evaluate_fitness_vector(AcyclicGraph &indv,
    TrainingData &train) {
  ImplicitTrainingData* temp = dynamic_cast<ImplicitTrainingData*>(&train);
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> deriv = indv.evaluate_deriv(
        temp->x);
  Eigen::ArrayXXd dot(deriv.second.rows(), deriv.second.cols());
  double infinity = std::numeric_limits<double>::infinity();

  if (normalize_dot) {
    dot = (deriv.second / (deriv.second.square().rowwise().sum().sqrt())) *
          (temp->dx_dt / (temp->dx_dt.square().rowwise().sum().sqrt()));

  } else {
    dot = deriv.second * temp->dx_dt;
  }

  if (required_params != 0) {
    int n_params_used;

    for (int i = 0; i < dot.rows(); ++i) {
      n_params_used = 0;

      for (int j = 0; j < dot.cols(); ++j) {
        if (dot(i, j) > 0) {
          ++n_params_used;
        }
      }

      if (n_params_used >= required_params) {
        return Eigen::ArrayXXd::Constant(deriv.second.rows(), 1, infinity);
      }
    }
  }

  Eigen::ArrayXXd fit(deriv.second.rows(), 1);
  fit = dot.rowwise().sum() / dot.abs().rowwise().sum();
  return fit;
}