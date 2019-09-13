#ifndef BINGOCCP_INCLUDE_BINGOCPP_FITNESS_FUNCTION_H_
#define BINGOCCP_INCLUDE_BINGOCPP_FITNESS_FUNCTION_H_

#include <stdexcept>
#include <string>
#include <unordered_set>

#include <BingoCpp/agraph.h>
#include <BingoCpp/training_data.h>

namespace bingo {

std::unordered_set<std::string> kMeanAbsoluteError;
std::unordered_set<std::string> kMeanSquaredError;
std::unordered_set<std::string> kRootMeanSquaredError;

class FitnessFunction {
 public:
  inline FitnessFunction(TrainingData* training_data = nullptr) :
    eval_count_(0), training_data_(training_data) { }
  virtual double EvaluateIndividualFitness(AGraph& individual) = 0;
 private:
  int eval_count_;
  TrainingData* training_data_;
};

inline bool metric_found(const std::unordered_set<std::string>& set,
                  std::string metric) {
  return set.find(metric) != set.end();
}

class VectorBasedFunction : public FitnessFunction {
 public:
  inline VectorBasedFunction(
      TrainingData* training_data = nullptr,
      std::string metric = "mae") : FitnessFunction(training_data) {
    if (metric_found(kMeanAbsoluteError, metric)) {
      metric_function_ = &VectorBasedFunction::mean_absolute_error;
    } else if (metric_found(kMeanSquaredError, metric)) {
      metric_function_ = &VectorBasedFunction::mean_squared_error;
    } else if (metric_found(kRootMeanSquaredError, metric)) {
      metric_function_ = &VectorBasedFunction::root_mean_square_error;
    } else {
      throw std::invalid_argument("Invalid metric for Fitness Function");
    }
  }

  inline double EvaluateIndividualFitness(AGraph& individual) {
    Eigen::ArrayXXd fitness_vector = EvaluateFitnessVector(individual);
    return (this->*metric_function_)(fitness_vector);
  }

  virtual const Eigen::ArrayXXd& EvaluateFitnessVector(AGraph& individual) = 0;

 protected:
  inline double mean_absolute_error(
      const Eigen::ArrayXXd& fitness_vector) {
    return fitness_vector.abs().mean();
  }

  inline double root_mean_square_error(
      const Eigen::ArrayXXd& fitness_vector) {
    return sqrt(fitness_vector.square().mean());
  }

  inline double mean_squared_error(
      const Eigen::ArrayXXd& fitness_vector) {
    return fitness_vector.square().mean();
  }

 private:
  double (VectorBasedFunction::*metric_function_)(const Eigen::ArrayXXd&);
};

std::unordered_set<std::string> kMeanAbsoluteError = {
  "mean_absolute_error",
  "mae"
};

std::unordered_set<std::string> kMeanSquaredError = {
  "mean_squared_error",
  "mse"
};

std::unordered_set<std::string> kRootMeanSquaredError = {
  "root_mean_squared_error",
  "rmse"
};
} // namespace bingo

#endif // BINGOCCP_INCLUDE_BINGOCPP_FITNESS_FUNCTION_H_