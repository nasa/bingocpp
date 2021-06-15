#include <Eigen/Dense>
#include <BingoCpp/gradient_mixin.h>

namespace bingo {

Eigen::ArrayXXd VectorGradientMixin::GetGradient(const Equation &individual) const {
  Eigen::ArrayXXd fitness_vector = this->EvaluateFitnessVector(individual);
  Eigen::ArrayXXd fitness_partials = this->GetJacobian(individual);
  return metric_derivative_(fitness_vector, fitness_partials.transpose());
}

} // namespace bingo
