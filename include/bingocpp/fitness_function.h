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
#ifndef BINGOCPP_INCLUDE_BINGOCPP_FITNESS_FUNCTION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_FITNESS_FUNCTION_H_

#include <stdexcept>
#include <string>
#include <unordered_set>

#include "bingocpp/agraph.h"
#include "bingocpp/equation.h"
#include "bingocpp/training_data.h"

namespace bingo {

const std::unordered_set<std::string> kMeanAbsoluteError = {
  "mean_absolute_error",
  "mae"
};

const std::unordered_set<std::string> kMeanSquaredError = {
  "mean_squared_error",
  "mse"
};

const std::unordered_set<std::string> kRootMeanSquaredError = {
  "root_mean_squared_error",
  "rmse"
};

class FitnessFunction {
 public:
  inline FitnessFunction(TrainingData *training_data = nullptr) :
    eval_count_(0), training_data_(training_data) { }

  virtual ~FitnessFunction() { }

  virtual double EvaluateIndividualFitness(Equation &individual) const = 0;

  int GetEvalCount() const {
    return eval_count_;
  }

  void SetEvalCount(int eval_count) {
    eval_count_ = eval_count;
  }

 protected:
  mutable int eval_count_;
  TrainingData* training_data_;
};

inline bool metric_found(const std::unordered_set<std::string> &set,
                         std::string metric) {
  return set.find(metric) != set.end();
}

class VectorBasedFunction : public FitnessFunction {
 public:
  typedef double (VectorBasedFunction::* MetricFunctionPointer)(const Eigen::ArrayXXd&);
  VectorBasedFunction(TrainingData *training_data = nullptr,
                      std::string metric = "mae") :
      FitnessFunction(training_data) {
    metric_function_ = GetMetric(metric);
  }

  virtual ~VectorBasedFunction() { }

  double EvaluateIndividualFitness(Equation &individual) const {
    Eigen::VectorXd fitness_vector = EvaluateFitnessVector(individual);
    return (const_cast<VectorBasedFunction*>(this)->*metric_function_)(fitness_vector);
  }

  virtual Eigen::VectorXd
  EvaluateFitnessVector(Equation &individual) const = 0;

 protected:
  double mean_absolute_error(const Eigen::ArrayXXd &fitness_vector) {
    return fitness_vector.abs().mean();
  }

  double root_mean_square_error(const Eigen::ArrayXXd &fitness_vector) {
    return sqrt(fitness_vector.square().mean());
  }

  double mean_squared_error(const Eigen::ArrayXXd &fitness_vector) {
    return fitness_vector.square().mean();
  }

  MetricFunctionPointer GetMetric(std::string metric) {
    if (metric_found(kMeanAbsoluteError, metric)) {
      return &VectorBasedFunction::mean_absolute_error;
    } else if (metric_found(kMeanSquaredError, metric)) {
      return &VectorBasedFunction::mean_squared_error;
    } else if (metric_found(kRootMeanSquaredError, metric)) {
      return &VectorBasedFunction::root_mean_square_error;
    } else {
      throw std::invalid_argument("Invalid metric for Fitness Function");
    }
  }

 private:
  mutable MetricFunctionPointer metric_function_;
};
} // namespace bingo

#endif // BINGOCPP_INCLUDE_BINGOCPP_FITNESS_FUNCTION_H_