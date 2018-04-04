/*!
 * \file fitness_tests.cc
 *
 * \author Ethan Adams
 * \date
 *
 * This file contains the unit tests for the functions associated with the
 * FitnessMetric class.
 */

#include <iostream>

#include "gtest/gtest.h"
#include "BingoCpp/fitness_metric.h"
#include "BingoCpp/acyclic_graph_nodes.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/NonLinearOptimization>


TEST(FitnessTest, optimize_constants) {
    StandardRegression sr;
    AcyclicGraph indv;
    Eigen::ArrayX3d stack2(12, 3);
    Eigen::ArrayXXd x(3, 3);
    stack2 << 0, 0, 0,
              0, 1, 1,
              1, -1, -1,
              1, -1, -1,
              5, 3, 1,
              5, 3, 1,
              2, 4, 2,
              2, 4, 2,
              4, 6, 0,
              4, 5, 6,
              3, 7, 6,
              3, 8, 0;
    indv.stack = stack2;
    AcyclicGraphManipulator manip = AcyclicGraphManipulator(3, 12, 1);
    manip.simplify_stack(indv);
    x << 1., 4., 7., 2., 5., 8., 3., 6., 9.;
    Eigen::ArrayXXd y(3, 1);
    y << 4.64, 8.28, 11.42;


    Eigen::VectorXd temp_con(2);
    temp_con << 15.0, 2.0;
    indv.set_constants(temp_con);
    ExplicitTrainingData train(x, y);

    sr.optimize_constants(indv, train);

    ASSERT_NEAR(3.14, indv.constants[0], .001);
    ASSERT_NEAR(10.0, indv.constants[1], .001);
}


TEST(FitnessTest, evaluate_fitness_vector) {
    StandardRegression sr;
    AcyclicGraph indv;
    Eigen::ArrayX3d stack2(12, 3);
    Eigen::ArrayXXd x(3, 3);
    stack2 << 0, 0, 0,
              0, 1, 1,
              1, 0, 0,
              1, 1, 1,
              5, 3, 1,
              5, 3, 1,
              2, 4, 2,
              2, 4, 2,
              4, 6, 0,
              4, 5, 6,
              3, 7, 6,
              3, 8, 0;
    indv.stack = stack2;
    AcyclicGraphManipulator manip = AcyclicGraphManipulator(3, 12, 1);
    manip.simplify_stack(indv);
    Eigen::VectorXd temp_con(2);
    temp_con << 3.14, 10.0;
    indv.set_constants(temp_con); 
    x << 1., 4., 7., 2., 5., 8., 3., 6., 9.;
    Eigen::ArrayXXd y(3, 1);
    y << 4.64, 8.28, 11.42;

    ExplicitTrainingData ex = ExplicitTrainingData(x, y);

    Eigen::ArrayXXd f = sr.evaluate_fitness_vector(indv, ex);

    for (size_t i = 0; i < y.rows(); ++i) {
        ASSERT_NEAR(f(i), 0, .001);
    }
}

TEST(FitnessTest, evaluate_fitness) {
    StandardRegression sr;
    AcyclicGraph indv;
    Eigen::ArrayX3d stack2(12, 3);
    Eigen::ArrayXXd x(3, 3);
    stack2 << 0, 0, 0,
              0, 1, 1,
              1, -1, -1,
              1, -1, -1,
              5, 3, 1,
              5, 3, 1,
              2, 4, 2,
              2, 4, 2,
              4, 6, 0,
              4, 5, 6,
              3, 7, 6,
              3, 8, 0;
    indv.stack = stack2;
    AcyclicGraphManipulator manip = AcyclicGraphManipulator(3, 12, 1);
    manip.simplify_stack(indv);
    x << 1., 4., 7., 2., 5., 8., 3., 6., 9.;
    Eigen::ArrayXXd y(3, 1);
    y << 4.64, 8.28, 11.42;
    // Eigen::VectorXd temp_con(2);
    // temp_con << 3.14, 10.0;

    ExplicitTrainingData ex = ExplicitTrainingData(x, y);

    float metric;
    metric = sr.evaluate_fitness(indv, ex);
    
    ASSERT_NEAR(metric, 0, .001);
}