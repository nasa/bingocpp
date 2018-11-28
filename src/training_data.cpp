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

ExplicitTrainingData::ExplicitTrainingData(Eigen::ArrayXXd vx,
    Eigen::ArrayXXd vy) {
  x = vx;
  y = vy;
}

ExplicitTrainingData* ExplicitTrainingData:: get_item(std::list<int> items) {
  int i = 0;
  Eigen::ArrayXXd temp_x(items.size(), x.cols());
  Eigen::ArrayXXd temp_y(items.size(), y.cols());

  for (std::list<int>::iterator it = items.begin(); it != items.end(); ++it) {
    temp_x.block(i, 0, 1, x.cols()) = x.block(*it, 0, 1, x.cols());
    temp_y.block(i, 0, 1, y.cols()) = y.block(*it, 0, 1, y.cols());
    ++i;
  }

  ExplicitTrainingData* temp = new ExplicitTrainingData(temp_x, temp_y);
  return temp;
}

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

ImplicitTrainingData* ImplicitTrainingData:: get_item(std::list<int> items) {
  int i = 0;
  Eigen::ArrayXXd temp_x(items.size(), x.cols());
  Eigen::ArrayXXd temp_dx_dt(items.size(), dx_dt.cols());

  for (std::list<int>::iterator it = items.begin(); it != items.end(); ++it) {
    temp_x.block(i, 0, 1, x.cols()) = x.block(*it, 0, 1, x.cols());
    temp_dx_dt.block(i, 0, 1, dx_dt.cols()) = dx_dt.block(*it, 0, 1, dx_dt.cols());
    ++i;
  }

  ImplicitTrainingData* temp = new ImplicitTrainingData(temp_x, temp_dx_dt);
  return temp;
}