#include <Eigen/Dense>
#include <BingoCpp/gradient_mixin.h>

namespace bingo {

VectorGradientMixin::VectorGradientMixin(TrainingData *training_data, std::string metric) : VectorBasedFunction(
    training_data, metric) {
  if (metric_found(kMeanAbsoluteError, metric)) {
    metric_derivative_ = VectorGradientMixin::mean_absolute_error_derivative;
  } else if (metric_found(kMeanSquaredError, metric)) {
    metric_derivative_ = VectorGradientMixin::mean_squared_error_derivative;
  } else {
    metric_derivative_ = VectorGradientMixin::root_mean_squared_error_derivative;
  }
}

std::tuple<double, Eigen::ArrayXXd> VectorGradientMixin::GetIndividualFitnessAndGradient(const Equation &individual) const {
  Eigen::ArrayXXd fitness_vector = this->EvaluateFitnessVector(individual);
  Eigen::ArrayXXd fitness_partials = this->GetJacobian(individual);
  return std::tuple<double, Eigen::ArrayXXd>{0.0, metric_derivative_(fitness_vector, fitness_partials.transpose())};
}

} // namespace bingo
