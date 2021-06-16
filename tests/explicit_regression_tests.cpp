#include <cmath>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <BingoCpp/explicit_regression.h>

#include "test_fixtures.h"

using namespace bingo;

namespace {

class TestExplicitRegression : public testing::Test {
 public:
  ExplicitTrainingData* training_data_;
  testutils::SumEquation sum_equation_;

  void SetUp() {
    training_data_ = init_sample_training_data();
    sum_equation_ = testutils::init_sum_equation();
  }

  void TearDown() {
    delete training_data_;
  }

 private:
  ExplicitTrainingData* init_sample_training_data() {
    const int num_points = 50;
    const int num_data_per_feature = 10;
    const int num_feature = 50 / num_data_per_feature;
    Eigen::ArrayXXd x = Eigen::ArrayXd::LinSpaced(num_points, 0, 0.98);
    x = x.reshaped(num_feature, num_data_per_feature);
    x.transposeInPlace();
    Eigen::Array<double, 10, 1> y = Eigen::ArrayXd::LinSpaced(10, 0.2, 4.7);
    return new ExplicitTrainingData(x, y);
  }
};

TEST_F(TestExplicitRegression, EvaluateIndividualFitness) {
  ExplicitRegression regressor(training_data_);
  double fitness = regressor.EvaluateIndividualFitness(sum_equation_);
  ASSERT_TRUE(fitness < 1e-10);
}

TEST_F(TestExplicitRegression, EvaluateIndividualFitnessWithNaN) {
  training_data_->x(0, 0) = std::numeric_limits<double>::quiet_NaN();
  ExplicitRegression regressor(training_data_);
  double fitness = regressor.EvaluateIndividualFitness(sum_equation_);
  ASSERT_TRUE(std::isnan(fitness));
}

TEST_F(TestExplicitRegression, GetFitnessVectorAndJacobian) {
  ExplicitRegression regressor(training_data_);
  Eigen::ArrayXXd expected_fitness_vector = Eigen::ArrayXXd::Zero(10, 1);
  Eigen::ArrayXXd expected_jacobian = training_data_->x;

  Eigen::ArrayXXd fitness_vector, jacobian;
  std::tie(fitness_vector, jacobian) = regressor.GetFitnessVectorAndJacobian(sum_equation_);

  ASSERT_TRUE(expected_fitness_vector.isApprox(fitness_vector));
  ASSERT_TRUE(expected_jacobian.isApprox(jacobian));
}

// TODO
//TEST_F(TestExplicitRegression, GetGradient) {
//
//}

TEST_F(TestExplicitRegression, GetSubsetOfTrainingData) {
  Eigen::ArrayXXd data_input = Eigen::ArrayXd::LinSpaced(5, 0, 4);
  ExplicitTrainingData* training_data = new ExplicitTrainingData(data_input, data_input);
  ExplicitTrainingData* subset_training_data = training_data->GetItem(std::vector<int>{0, 2, 3});

  Eigen::ArrayXXd expected_subset(3, 1);
  expected_subset << 0, 2, 3;
  ASSERT_TRUE(subset_training_data->x.isApprox(expected_subset));
  ASSERT_TRUE(subset_training_data->y.isApprox(expected_subset));
  delete training_data,
  delete subset_training_data;
}

TEST_F(TestExplicitRegression, CorrectTrainingDataSize) {
  for (int size : std::vector<int> {2, 5, 50}) {
    Eigen::ArrayXXd data_input = Eigen::ArrayXd::LinSpaced(size, 0, 10);
    ExplicitTrainingData* training_data = new ExplicitTrainingData(data_input, data_input);
    ASSERT_EQ(training_data->Size(), size);
    delete training_data;
  }
}
} // namespace 