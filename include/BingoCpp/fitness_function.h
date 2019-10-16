#ifndef BINGOCPP_INCLUDE_BINGOCPP_FITNESS_FUNCTION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_FITNESS_FUNCTION_H_

#include <stdexcept>
#include <string>
#include <unordered_set>

#include "BingoCpp/agraph.h"
#include "BingoCpp/equation.h"
#include "BingoCpp/training_data.h"

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

  virtual double EvaluateIndividualFitness(const Equation &individual) const = 0;

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

  double EvaluateIndividualFitness(const Equation &individual) const {
    Eigen::ArrayXXd fitness_vector = EvaluateFitnessVector(individual);
    return (const_cast<VectorBasedFunction*>(this)->*metric_function_)(fitness_vector);
  }

  virtual Eigen::ArrayXXd
  EvaluateFitnessVector(const Equation &individual) const = 0;

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