#include <cmath>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <BingoCpp/implicit_regression.h>

#include "test_fixtures.h"

using namespace bingo;

namespace {

class TestImplicitRegression : public testing::Test {
 public:
  ImplicitTrainingData *training_data_;
  testutils::SumEquation sum_equation_;

  void SetUp() {
    training_data_ = init_sample_training_data();
    sum_equation_ = testutils::init_sum_equation();
  }

  void TearDown() {
    delete training_data_;
  }

 private:
  ImplicitTrainingData* init_sample_training_data() {
    const int num_points = 50;
    const int num_data_per_feature = 10;
    const int num_feature = 50 / num_data_per_feature;
    const int index_block_mod = 3;

    Eigen::ArrayXXd x = Eigen::ArrayXd::LinSpaced(num_points, 0, 0.98);
    x = x.reshaped(num_feature, num_data_per_feature);
    x.transposeInPlace();

    Eigen::ArrayXXd dx_dt = Eigen::ArrayXXd::Constant(x.rows(), x.cols(), 1);
    dx_dt.block(
        0, index_block_mod, dx_dt.rows(), dx_dt.cols() - index_block_mod)
        = Eigen::ArrayXXd::Constant(x.rows(), 2, -1);
    dx_dt.col(dx_dt.cols()/2) = Eigen::ArrayXd::Constant(dx_dt.rows(), 0);
    return new ImplicitTrainingData(x, dx_dt);
  }
};

TEST_F(TestImplicitRegression, EvaluateFinessIndividual) {
  ImplicitRegression *regressor = new ImplicitRegression(training_data_, -1, true);
  double fitness = regressor->EvaluateIndividualFitness(sum_equation_);
  ASSERT_TRUE(0.14563031020 - fitness < 1e-10);
  delete regressor;
}

TEST_F(TestImplicitRegression, GetSubsetOfData) {
  auto data_input = Eigen::ArrayXd::LinSpaced(5, 0, 4);
  auto training_data = new ImplicitTrainingData(data_input, data_input);
  auto subset_training_data = training_data->GetItem(std::vector<int>{0, 2, 3});
  Eigen::ArrayXXd expected_subset(3, 1);
  expected_subset << 0, 2, 3;
  ASSERT_TRUE(subset_training_data->x.isApprox(expected_subset));
  ASSERT_TRUE(subset_training_data->dx_dt.isApprox(expected_subset));
  delete training_data, subset_training_data;
}

TEST_F(TestImplicitRegression, CorrectTrainingDataSize) {
  for (int size : std::vector<int> {2, 5, 50}) {
    Eigen::ArrayXXd data_input = Eigen::ArrayXd::LinSpaced(size, 0, 10);
    auto training_data = new ImplicitTrainingData(data_input, data_input);
    ASSERT_EQ(training_data->Size(), size);
    delete training_data;
  }
}
} // namespace (anonymous)