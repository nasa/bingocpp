#include <iostream>

#include "bingocpp/explicit_regression.h"

namespace bingo {

ExplicitTrainingData *ExplicitTrainingData::GetItem(int item) {
  return new ExplicitTrainingData(x.row(item), y.row(item));
}

ExplicitTrainingData *ExplicitTrainingData::GetItem(
    const std::vector<int> &items) {
  Eigen::ArrayXXd temp_in(items.size(), x.cols());
  Eigen::ArrayXXd temp_out(items.size(), y.cols());

  for (unsigned int row = 0; row < items.size(); row ++) {
    temp_in.row(row) = x.row(items[row]);
    temp_out.row(row) = y.row(items[row]);
  }

  return new ExplicitTrainingData(temp_in, temp_out);
}

Eigen::VectorXd ExplicitRegression::EvaluateFitnessVector(
    Equation &individual) const {
  ++ eval_count_;
  const Eigen::ArrayXXd x = ((ExplicitTrainingData*)training_data_)->x;
  Eigen::ArrayXXd f_of_x = individual.EvaluateEquationAt(x);
  return f_of_x - ((ExplicitTrainingData*)training_data_)->y;
}

ExplicitRegressionState ExplicitRegression::DumpState() {
  return ExplicitRegressionState(
          ((ExplicitTrainingData*)training_data_)->DumpState(),
          metric_, eval_count_);
}


} // namespace bingo