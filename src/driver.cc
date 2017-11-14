/*!
 * \file driver.cc
 *
 * \author Geoffrey F. Bomarito
 * \date
 *
 * This file contains the main function for BingoCpp.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Eigen/Dense>

#include <iostream>
#include <chrono>

#include "BingoCpp/version.h"
#include "BingoCpp/acyclic_graph.h"


int test_eig() {
  Eigen::MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;
}

void TestAcyclicGraph(int num_loops, int num_evals) {
  CommandStack stack = CommandStack();
  Eigen::ArrayXXd x(60, 3);
  std::vector<double> constants = std::vector<double>();
  // y = x_0 * ( C_0 + C_1/x_1 ) - x_0
  stack.push_back(std::make_pair(0, std::vector<int>()));
  stack[0].second.push_back(0);
  stack.push_back(std::make_pair(0, std::vector<int>()));
  stack[1].second.push_back(1);
  stack.push_back(std::make_pair(1, std::vector<int>()));
  stack[2].second.push_back(0);
  stack.push_back(std::make_pair(1, std::vector<int>()));
  stack[3].second.push_back(1);
  stack.push_back(std::make_pair(5, std::vector<int>()));
  stack[4].second.push_back(3);
  stack[4].second.push_back(1);
  stack.push_back(std::make_pair(5, std::vector<int>()));
  stack[5].second.push_back(3);
  stack[5].second.push_back(1);
  stack.push_back(std::make_pair(2, std::vector<int>()));
  stack[6].second.push_back(4);
  stack[6].second.push_back(2);
  stack.push_back(std::make_pair(2, std::vector<int>()));
  stack[7].second.push_back(4);
  stack[7].second.push_back(2);
  stack.push_back(std::make_pair(4, std::vector<int>()));
  stack[8].second.push_back(6);
  stack[8].second.push_back(0);
  stack.push_back(std::make_pair(4, std::vector<int>()));
  stack[9].second.push_back(5);
  stack[9].second.push_back(6);
  stack.push_back(std::make_pair(3, std::vector<int>()));
  stack[10].second.push_back(7);
  stack[10].second.push_back(6);
  stack.push_back(std::make_pair(3, std::vector<int>()));
  stack[11].second.push_back(8);
  stack[11].second.push_back(0);
  x << 1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9., 
       1., 2., 3., 4., 5., 6., 7., 8., 9., 1., 2., 3., 4., 5., 6., 7., 8., 9.;
  constants.push_back(3.14);
  constants.push_back(10.0);
  //PrintStack(stack);
  Eigen::ArrayXXd y;
  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> y_and_dy;

  double avg_time_per_eval = 0.;
  double avg_time_per_seval = 0.;
  double avg_time_per_deval = 0.;
  double avg_time_per_sdeval = 0.;
  for (int i = 0; i < num_loops; ++i) {
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double, std::micro> duration;
    
    t1 =std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_evals; ++i) {
      y = Evaluate(stack, x, constants);
    }
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    avg_time_per_eval += duration.count()/num_evals;
    
    t1 =std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_evals; ++i) {
      y = SimplifyAndEvaluate(stack, x, constants);
    }
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    avg_time_per_seval += duration.count()/num_evals;
    
    t1 =std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_evals; ++i) {
      y_and_dy = EvaluateWithDerivative(stack, x, constants);
    }
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    avg_time_per_deval += duration.count()/num_evals;
    
    t1 =std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_evals; ++i) {
      y_and_dy = SimplifyAndEvaluateWithDerivative(stack, x, constants);
    }
    t2 = std::chrono::high_resolution_clock::now();
    duration = t2 - t1;
    avg_time_per_sdeval += duration.count()/num_evals;
  }
  avg_time_per_eval /= num_loops;
  avg_time_per_seval /= num_loops;
  avg_time_per_deval /= num_loops;
  avg_time_per_sdeval /= num_loops;
  std::cout << "Eval:              " << avg_time_per_eval << " microseconds\n";
  std::cout << "Simple Eval:       " << avg_time_per_seval << " microseconds\n";
  std::cout << "Eval Deriv:        " << avg_time_per_deval << " microseconds\n";
  std::cout << "Simple Eval Deriv: " << avg_time_per_sdeval << " microseconds\n";
}


int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stdout, "%s Version %d.%d\n", argv[0], Tutorial_VERSION_MAJOR,
            Tutorial_VERSION_MINOR);
    fprintf(stdout, "Usage: %s number number\n", argv[0]);
    return 1;
  }

  TestAcyclicGraph(std::atol(argv[1]), std::atol(argv[2]));
  return 0;
}






