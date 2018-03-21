/*!
 * \file training_data.cc
 *
 * \author Ethan Adams
 * \date
 *
 * This file contains the unit tests for TrainingData class
 */

#include <iostream>

#include "gtest/gtest.h"
#include "BingoCpp/training_data.h"
#include <Eigen/Dense>
#include <Eigen/Core>

TEST(TrainingDataTest, ExplicitConstruct) {
    Eigen::ArrayXXd x(4, 3);
    Eigen::ArrayXXd y(4, 2);
    x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 7;
    y << 6, 7, 1, 2, 4, 5, 8, 9;

    ExplicitTrainingData ex = ExplicitTrainingData(x, y);

    for (int i = 0; i < 4; ++i) {
        ASSERT_DOUBLE_EQ(x(i, 0), ex.x(i, 0));
        ASSERT_DOUBLE_EQ(x(i, 1), ex.x(i, 1));
        ASSERT_DOUBLE_EQ(x(i, 2), ex.x(i, 2));
        ASSERT_DOUBLE_EQ(y(i, 0), ex.y(i, 0));
        ASSERT_DOUBLE_EQ(y(i, 1), ex.y(i, 1));
    }
}

TEST(TrainingDataTest, ExplicitGetItem) {
    Eigen::ArrayXXd x(4, 3);
    Eigen::ArrayXXd y(4, 2);
    x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 7;
    y << 6, 7, 1, 2, 4, 5, 8, 9;
    std::list<int> items;
    items.push_back(1);
    items.push_back(3);

    ExplicitTrainingData ex = ExplicitTrainingData(x, y);
    ExplicitTrainingData* slice = ex.get_item(items);

    Eigen::ArrayXXd truth_x(2, 3);
    Eigen::ArrayXXd truth_y(2, 2);
    truth_x << 4, 5, 6, 7, 4, 7;
    truth_y << 1, 2, 8, 9;

    for (int i = 0; i < 2; ++i) {
        ASSERT_DOUBLE_EQ(truth_x(i, 0), slice->x(i, 0));
        ASSERT_DOUBLE_EQ(truth_x(i, 1), slice->x(i, 1));
        ASSERT_DOUBLE_EQ(truth_x(i, 2), slice->x(i, 2));
        ASSERT_DOUBLE_EQ(truth_y(i, 0), slice->y(i, 0));
        ASSERT_DOUBLE_EQ(truth_y(i, 1), slice->y(i, 1));
    }
}

TEST(TrainingDataTest, ExplicitSize) {
    Eigen::ArrayXXd x(4, 3);
    Eigen::ArrayXXd y(4, 2);
    x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 7;
    y << 6, 7, 1, 2, 4, 5, 8, 9;

    ExplicitTrainingData ex = ExplicitTrainingData(x, y);
    ASSERT_EQ(4, ex.size());
}

TEST(TrainingDataTest, ImplicitConstruct) {
    Eigen::ArrayXXd x(4, 3);
    Eigen::ArrayXXd dx_dt(4, 2);
    x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 7;
    dx_dt << 6, 7, 1, 2, 4, 5, 8, 9;

    ImplicitTrainingData im = ImplicitTrainingData(x, dx_dt);

    for (int i = 0; i < 4; ++i) {
        ASSERT_DOUBLE_EQ(x(i, 0), im.x(i, 0));
        ASSERT_DOUBLE_EQ(x(i, 1), im.x(i, 1));
        ASSERT_DOUBLE_EQ(x(i, 2), im.x(i, 2));
        ASSERT_DOUBLE_EQ(dx_dt(i, 0), im.dx_dt(i, 0));
        ASSERT_DOUBLE_EQ(dx_dt(i, 1), im.dx_dt(i, 1));
    }
}

TEST(TrainingDataTest, ImplicitGetItem) {
    Eigen::ArrayXXd x(4, 3);
    Eigen::ArrayXXd dx_dt(4, 2);
    x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 7;
    dx_dt << 6, 7, 1, 2, 4, 5, 8, 9;
    std::list<int> items;
    items.push_back(1);
    items.push_back(3);

    ImplicitTrainingData im = ImplicitTrainingData(x, dx_dt);
    ImplicitTrainingData* slice = im.get_item(items);

    Eigen::ArrayXXd truth_x(2, 3);
    Eigen::ArrayXXd truth_dx_dt(2, 2);
    truth_x << 4, 5, 6, 7, 4, 7;
    truth_dx_dt << 1, 2, 8, 9;

    for (int i = 0; i < 2; ++i) {
        ASSERT_DOUBLE_EQ(truth_x(i, 0), slice->x(i, 0));
        ASSERT_DOUBLE_EQ(truth_x(i, 1), slice->x(i, 1));
        ASSERT_DOUBLE_EQ(truth_x(i, 2), slice->x(i, 2));
        ASSERT_DOUBLE_EQ(truth_dx_dt(i, 0), slice->dx_dt(i, 0));
        ASSERT_DOUBLE_EQ(truth_dx_dt(i, 1), slice->dx_dt(i, 1));
    }
}

TEST(TrainingDataTest, ImplicitSize) {
    Eigen::ArrayXXd x(4, 3);
    Eigen::ArrayXXd dx_dt(4, 2);
    x << 1, 2, 3, 4, 5, 6, 7, 8, 9, 7, 4, 7;
    dx_dt << 6, 7, 1, 2, 4, 5, 8, 9;

    ImplicitTrainingData im = ImplicitTrainingData(x, dx_dt);

    ASSERT_EQ(4, im.size());
}