#include <Eigen/Dense>
#include <BingoCpp/gradient_mixin.h>

#include <tuple>

namespace bingo {

VectorGradientMixin::VectorGradientMixin(TrainingData *training_data, std::string metric) {
  if (metric_found(kMeanAbsoluteError, metric)) {
    metric_function_ = VectorGradientMixin::mean_absolute_error;
    metric_derivative_ = VectorGradientMixin::mean_absolute_error_derivative;
  } else if (metric_found(kMeanSquaredError, metric)) {
    metric_function_ = VectorGradientMixin::mean_squared_error;
    metric_derivative_ = VectorGradientMixin::mean_squared_error_derivative;
  } else if (metric_found(kRootMeanSquaredError, metric)) {
    metric_function_ = VectorGradientMixin::root_mean_squared_error;
    metric_derivative_ = VectorGradientMixin::root_mean_squared_error_derivative;
  } else {
    throw std::invalid_argument("Invalid metric for VectorGradientMixin");
  }
}

std::tuple<double, Eigen::ArrayXXd> VectorGradientMixin::GetIndividualFitnessAndGradient(const Equation &individual) const {
  Eigen::ArrayXXd fitness_vector, jacobian;
  std::tie(fitness_vector, jacobian) = this->GetFitnessVectorAndJacobian(individual);
  double fitness = this->metric_function_(fitness_vector);
  return std::tuple<double, Eigen::ArrayXXd>{fitness, metric_derivative_(fitness_vector, jacobian.transpose())};
}

} // namespace bingo
