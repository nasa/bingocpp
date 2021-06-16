#include <iostream>

#include "BingoCpp/explicit_regression.h"
#include <tuple>

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

Eigen::ArrayXXd ExplicitRegression::EvaluateFitnessVector(
    const Equation &individual) const {
  ++ eval_count_;
  const Eigen::ArrayXXd x = ((ExplicitTrainingData*)training_data_)->x;
  Eigen::ArrayXXd f_of_x = individual.EvaluateEquationAt(x);
  return f_of_x - ((ExplicitTrainingData*)training_data_)->y;
}

std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd> ExplicitRegression::GetFitnessVectorAndJacobian(
    const Equation &individual) const {
  Eigen::ArrayXXd f_of_x, df_dc;
  const Eigen::ArrayXXd x = ((ExplicitTrainingData*)training_data_)->x;
  std::tie(f_of_x, df_dc) = individual.EvaluateEquationWithLocalOptGradientAt(x);
  return std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXd>{this->EvaluateFitnessVector(individual), df_dc};
}

} // namespace bingo