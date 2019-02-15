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
void run_benchmarks(BenchMarkTestData &benchmark_test_data);
Eigen::ArrayXd time_benchmark(
  void (*benchmark)(std::vector<AGraphValues>&, Eigen::ArrayXXd&), 
  BenchMarkTestData &test_data, int number=100, int repeat=10);
void benchmark_evaluate(std::vector<AGraphValues> &indv_list,
                        Eigen::ArrayXXd &x_vals);
void benchmark_evaluate_w_x_derivative(std::vector<AGraphValues> &indv_list,
                                       Eigen::ArrayXXd &x_vals);
void benchmark_evaluate_w_c_derivative(std::vector<AGraphValues> &indv_list,
                                       Eigen::ArrayXXd &x_vals);
void print_results(Eigen::ArrayXd &run_times);
void print_header();
double max_val(Eigen::ArrayXd &run_times);
double min_val(Eigen::ArrayXd &run_times);
double standard_deviation(Eigen::ArrayXd &vec);

#endif