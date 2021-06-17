#include <Eigen/Dense>
#include <bingocpp/equation.h>
#include <bingocpp/fitness_function.h>

#include <functional>
#include <cmath>

namespace bingo {

class GradientMixin {
 public:
  virtual std::tuple<double, Eigen::VectorXd> GetIndividualFitnessAndGradient(Equation &individual) const = 0;
};

class VectorGradientMixin : public GradientMixin {
 protected:
  static double mean_absolute_error(const Eigen::ArrayXXd &fitness_vector) {
    return fitness_vector.abs().mean();
  }

  static double mean_squared_error(const Eigen::ArrayXXd &fitness_vector) {
    return fitness_vector.square().mean();
  }

  static double root_mean_squared_error(const Eigen::ArrayXXd &fitness_vector) {
    return sqrt(fitness_vector.square().mean());
  }

  static Eigen::VectorXd mean_absolute_error_derivative(const Eigen::VectorXd &fitness_vector, const Eigen::ArrayXXd &fitness_partials) {
    return (fitness_partials.rowwise() * fitness_vector.transpose()(0, Eigen::all).array().sign()).rowwise().mean();
  }

  static Eigen::VectorXd mean_squared_error_derivative(const Eigen::VectorXd &fitness_vector, const Eigen::ArrayXXd &fitness_partials) {
    return 2.0 * (fitness_partials.rowwise() * fitness_vector.transpose()(0, Eigen::all).array()).rowwise().mean();
  }

  static Eigen::VectorXd root_mean_squared_error_derivative(const Eigen::VectorXd &fitness_vector, const Eigen::ArrayXXd &fitness_partials) {
    return 1.0/sqrt(fitness_vector.transpose()(0, Eigen::all).array().square().mean()) *
    (fitness_partials.rowwise() * fitness_vector.transpose()(0, Eigen::all).array()).rowwise().mean();
  }

 private:
  std::function<double(Eigen::ArrayXXd)> metric_function_;
  std::function<Eigen::ArrayXXd(Eigen::ArrayXXd, Eigen::ArrayXXd)> metric_derivative_;

 public:
  VectorGradientMixin(TrainingData *training_data = nullptr, std::string metric = "mae");

  std::tuple<double, Eigen::VectorXd> GetIndividualFitnessAndGradient(Equation &individual) const;

  virtual std::tuple<Eigen::VectorXd, Eigen::ArrayXXd> GetFitnessVectorAndJacobian(Equation &individual) const = 0;
};

} // namespace bingo
