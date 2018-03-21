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

ExplicitTrainingData::ExplicitTrainingData(Eigen::ArrayXXd vx, Eigen::ArrayXXd vy) {
    x = vx;
    y = vy;
}

ExplicitTrainingData* ExplicitTrainingData:: get_item(std::list<int> items) {
    int i = 0;
    Eigen::ArrayXXd temp_x(items.size(), x.cols());
    Eigen::ArrayXXd temp_y(items.size(), y.cols());
    for (std::list<int>::iterator it=items.begin(); it != items.end(); ++it) {
        temp_x.block<1, 3>(i, 0) = x.block<1, 3>(*it, 0); 
        temp_y.block<1, 3>(i, 0) = y.block<1, 3>(*it, 0);
        // temp_x.block<1, x.cols()>(i, 0) = x.block<1, x.cols()>(*it, 0); 
        // temp_y.block<1, y.cols()>(i, 0) = y.block<1, y.cols()>(*it, 0);
        ++i;
    }
    ExplicitTrainingData* temp = new ExplicitTrainingData(temp_x, temp_y);
    return temp;
}

ImplicitTrainingData::ImplicitTrainingData(Eigen::ArrayXXd vx, Eigen::ArrayXXd vdx_dt) {
    x = vx;
    dx_dt = vdx_dt;
}

ImplicitTrainingData* ImplicitTrainingData:: get_item(std::list<int> items) {
    int i = 0;
    Eigen::ArrayXXd temp_x(items.size(), x.cols());
    Eigen::ArrayXXd temp_dx_dt(items.size(), dx_dt.cols());
    for (std::list<int>::iterator it=items.begin(); it != items.end(); ++it) {
        temp_x.block<1, 3>(i, 0) = x.block<1, 3>(*it, 0); 
        temp_dx_dt.block<1, 3>(i, 0) = dx_dt.block<1, 3>(*it, 0);
        ++i;
    }
    ImplicitTrainingData* temp = new ImplicitTrainingData(temp_x, temp_dx_dt);
    return temp;
}