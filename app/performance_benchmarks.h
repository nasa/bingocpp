#ifndef _PERFORMANCE_BENCHMARKS_H_
#define _PERFORMANCE_BENCHMARKS_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <istream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <Eigen/Core>

#include "BingoCpp/acyclic_graph.h"

#define EVALUATE "pure c++: evaluate"
#define X_DERIVATIVE "pure c++: x derivative"
#define C_DERIVATIVE "pure c++: c derivative"
#define STACK_FILE "test-agraph-stacks.csv"
#define CONST_FILE "test-agraph-consts.csv"
#define X_FILE "test-agraph-x-vals.csv"
#define INPUT_DIM	4
#define NUM_DATA_POINTS 128 
#define STACK_SIZE 128
#define STACK_COLS 3

struct AGraphValues {
  Eigen::ArrayX3i command_array;
  Eigen::VectorXd constants;
};

struct BenchMarkTestData {
  std::vector<AGraphValues> indv_list;
  Eigen::ArrayXXd x_vals;
  BenchMarkTestData() {}
  BenchMarkTestData(std::vector<AGraphValues> &il, Eigen::ArrayXXd &x):
    indv_list(il), x_vals(x) {}
};

void do_benchmarking();
void load_benchmark_data(BenchMarkTestData &benchmark_test_data);
void load_agraph_indvidual_data(std::vector<AGraphValues> &indv_list);
void set_indv_constants(AGraphValues &indv, std::string &const_string);
void set_indv_stack(AGraphValues &indv, std::string &stack_string);
Eigen::ArrayXXd load_agraph_x_vals();
void run_benchmarks(const BenchMarkTestData &benchmark_test_data);
Eigen::ArrayXd time_benchmark(
  void (*benchmark)(const std::vector<AGraphValues>&, const Eigen::ArrayXXd&), 
  const BenchMarkTestData &test_data, int number=100, int repeat=10);
void benchmark_evaluate(const std::vector<AGraphValues> &indv_list,
                        const Eigen::ArrayXXd &x_vals);
void benchmark_evaluate_w_x_derivative(const std::vector<AGraphValues> &indv_list,
                                       const Eigen::ArrayXXd &x_vals);
void benchmark_evaluate_w_c_derivative(const std::vector<AGraphValues> &indv_list,
                                       const Eigen::ArrayXXd &x_vals);
void print_header();
void print_results(const Eigen::ArrayXd &run_times, const std::string &name);
std::string string_precision(double val, int precision);
void output_params(const std::string &name, const std::string &mean, 
                   const std::string &std, const std::string &min, 
                   const std::string &max); 
double standard_deviation(const Eigen::ArrayXd &vec);

#endif