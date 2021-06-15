#include <Eigen/Dense>
#include <BingoCpp/equation.h>
#include <BingoCpp/fitness_function.h>

#include <functional>
#include <cmath>

namespace bingo {

class GradientMixin {
 public:
  virtual Eigen::ArrayXXd GetGradient(const Equation &individual) const = 0;
};

class VectorGradientMixin : public GradientMixin, public VectorBasedFunction {
 protected:
  static Eigen::ArrayXXd mean_absolute_error_derivative(const Eigen::ArrayXXd &fitness_vector, const Eigen::ArrayXXd &fitness_partials) {
    return (fitness_partials.rowwise() * fitness_vector(0, Eigen::all).sign()).rowwise().mean().transpose();
  }

  static Eigen::ArrayXXd mean_squared_error_derivative(const Eigen::ArrayXXd &fitness_vector, const Eigen::ArrayXXd &fitness_partials) {
    return 2.0 * (fitness_partials.rowwise() * fitness_vector(0, Eigen::all)).rowwise().mean().transpose();
  }

  static Eigen::ArrayXXd root_mean_squared_error_derivative(const Eigen::ArrayXXd &fitness_vector, const Eigen::ArrayXXd &fitness_partials) {
    return 1.0/sqrt(fitness_vector(0, Eigen::all).square().mean()) * (fitness_partials.rowwise() * fitness_vector(0, Eigen::all)).rowwise().mean().transpose();
  }

 private:
  std::function<Eigen::ArrayXXd(Eigen::ArrayXXd, Eigen::ArrayXXd)> metric_derivative_;

 public:
  Eigen::ArrayXXd GetGradient(const Equation &individual) const;

  virtual Eigen::ArrayXXd GetJacobian(const Equation &individual) const = 0;

  VectorGradientMixin(TrainingData *training_data = nullptr, std::string metric = "mae") : VectorBasedFunction(training_data, metric) {
    if (metric_found(kMeanAbsoluteError, metric)) {
      metric_derivative_ = VectorGradientMixin::mean_absolute_error_derivative;
    } else if (metric_found(kMeanSquaredError, metric)) {
      metric_derivative_ = VectorGradientMixin::mean_squared_error_derivative;
    } else {
      metric_derivative_ = VectorGradientMixin::root_mean_squared_error_derivative;
    }
  }
};

} // namespace bingo
