#include <gtest/gtest.h>
#include <iostream>

#include <Eigen/Dense>

#include <BingoCpp/gradient_mixin.h>
#include <BingoCpp/equation.h>
#include <BingoCpp/fitness_function.h>
#include <BingoCpp/AGraph.h>

#include <tuple>
#include <string>
#include <vector>
#include <cmath>

using namespace bingo;

namespace {

//class VectorGradFitnessFunction : public VectorGradientMixin, public VectorBasedFunction {
class VectorGradFitnessFunction : public VectorGradientMixin {
 public:
  // TODO figure out constructor
  VectorGradFitnessFunction(TrainingData *training_data = nullptr, std::string metric = "mae") : VectorGradientMixin(training_data, metric) {
  }

  AGraph empty_individual_;
  Eigen::ArrayXXd EvaluateFitnessVector(const Equation &individual) const {
    Eigen::ArrayXXd fitnessVector(1, 3);
    fitnessVector << -2.0, 0.0, 2.0;
    return fitnessVector;
  }

  Eigen::ArrayXXd GetJacobian(const Equation &individual) const {
    Eigen::ArrayXXd jacobian(3, 2);
    jacobian << 0.5, 1.0,
                1.0, 2.0,
               -0.5, 3.0;
    return jacobian;
  }
};

class GradientMixinTest: public ::testing::TestWithParam<std::tuple<std::string, std::vector<double>>> {
 public:
  VectorGradFitnessFunction fitness_function_; 
  std::string metric_;
  Eigen::ArrayXXd expected_gradient_;

  void SetUp() {
    std::tie(fitness_function_metric_, expected_gradient_data_) = GetParam();
    expected_gradient_ = Eigen::ArrayXXd(1, 2);
    expected_gradient_ << expected_gradient_data_[0], expected_gradient_data_[1];
    fitness_function_ = VectorGradFitnessFunction(nullptr, fitness_function_metric_);
  }

  void TearDown() { }

 private:
  std::string fitness_function_metric_;
  std::vector<double> expected_gradient_data_;
};


TEST_P(GradientMixinTest, VectorGradient) {
  ASSERT_TRUE(expected_gradient_.isApprox(fitness_function_.GetGradient(fitness_function_.empty_individual_)));
}


INSTANTIATE_TEST_SUITE_P(VectorGradientWithMetrics,
                         GradientMixinTest,
                         ::testing::Values(
                         std::make_tuple("mae", std::vector<double> {-1.0/3.0, 2.0/3.0}),
                         std::make_tuple("mse", std::vector<double> {-4.0/3.0, 8.0/3.0}),
                         std::make_tuple("rmse", std::vector<double> {sqrt(3.0/8.0) * -2.0/3.0, sqrt(3.0/8.0) * 4.0/3.0})));

TEST(TestGradientMixin, InvalidGradientMetric) {
  try {
    VectorGradFitnessFunction test_function(nullptr, "invalid_metric");
    FAIL() << "Expecting std::invalid_argument exception\n";
  } catch (std::invalid_argument &exception) {
    SUCCEED();
  }
}

} // namespace
