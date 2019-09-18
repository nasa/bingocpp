/*!
 * \file training_data.cc
 *
 * \author Ethan Adams
 * \date
 *
 * This file contains the cpp version of TrainingData.py
 */

#include "BingoCpp/training_data.h"
#include <iostream>
#include <stdlib.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "BingoCpp/utils.h"

namespace bingo {

namespace {
TrainingData* copy_data(const std::vector<int>& items,
                        const Eigen::ArrayXXd& input,
                        const Eigen::ArrayXXd& output,
                        bool implicit = false);
} // namespace (anonymous)

ImplicitTrainingData::ImplicitTrainingData(Eigen::ArrayXXd vx) {
  std::vector<Eigen::ArrayXXd> temp = calculate_partials(vx);
  x = temp[0];
  dx_dt = temp[1];
}

ImplicitTrainingData::ImplicitTrainingData(Eigen::ArrayXXd vx,
    Eigen::ArrayXXd vdx_dt) {
  x = vx;
  dx_dt = vdx_dt;
}

ImplicitTrainingData* ImplicitTrainingData::GetItem(int item) {
  return new ImplicitTrainingData(x.row(item), dx_dt.row(item));
}

ImplicitTrainingData* ImplicitTrainingData::GetItem(
    const std::vector<int>& items) {
  return (ImplicitTrainingData*)copy_data(items, x, dx_dt, true);
}

namespace {
TrainingData* copy_data(const std::vector<int>& items,
                        const Eigen::ArrayXXd& input,
                        const Eigen::ArrayXXd& output,
                        bool implicit) {
  Eigen::ArrayXXd temp_in(items.size(), input.cols());
  Eigen::ArrayXXd temp_out(items.size(), output.cols());

  for (int row = 0; row < items.size(); row ++) {
    temp_in.block(row, 0, 1, input.cols()) = input.block(items[row], 0, 1, input.cols());
    temp_out.block(row, 0, 1, output.cols()) = output.block(items[row], 0, 1, output.cols());
  }
  return (TrainingData*)new ImplicitTrainingData(temp_in, temp_out);
}
} // namespace (anonymous)
} // namespace bingo