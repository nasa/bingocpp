<<<<<<< HEAD
#include "performance_benchmarks.h"
=======
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
>>>>>>> 6ff719daae9de5c48dffdc2bbcad52ab16aa43e5

int main() {
  do_benchmarking();
  return 0;
}

void do_benchmarking() {
  BenchMarkTestData benchmark_test_data =  BenchMarkTestData();
  load_benchmark_data(benchmark_test_data);
  run_benchmarks(benchmark_test_data);
}

void load_benchmark_data(BenchMarkTestData &benchmark_test_data) {
  std::vector<AGraphValues> indv_list= std::vector<AGraphValues>();
  load_agraph_indvidual_data(indv_list);
  Eigen::ArrayXXd x_vals = load_agraph_x_vals();
  benchmark_test_data = BenchMarkTestData(indv_list, x_vals);
}

void load_agraph_indvidual_data(std::vector<AGraphValues> &indv_list) {
  std::ifstream stack_filestream;
  std::ifstream const_filestream;
  stack_filestream.open(STACK_FILE);
  const_filestream.open(CONST_FILE);

  std::string stack_file_line;
  std::string const_file_line;
  while ((stack_filestream >> stack_file_line) && 
      (const_filestream >> const_file_line)) {
    AGraphValues curr_indv = AGraphValues();
    set_indv_stack(curr_indv, stack_file_line);
    set_indv_constants(curr_indv, const_file_line);
    indv_list.push_back(curr_indv);
  }
  stack_filestream.close();
  const_filestream.close();
}

void set_indv_constants(AGraphValues &indv, std::string &const_string) {
  std::stringstream string_stream(const_string);
  std::string num_constants;
  std::getline(string_stream, num_constants, ',');
  Eigen::VectorXd curr_const = Eigen::VectorXd(std::stoi(num_constants));

  std::string curr_val;
  for (int i=0; std::getline(string_stream, curr_val, ','); i++) {
    curr_const(i) = std::stod(curr_val);
  }
  indv.constants = curr_const;
}

void set_indv_stack(AGraphValues &indv, std::string &stack_string) {
  std::stringstream string_stream(stack_string);
  Eigen::ArrayX3i curr_stack = Eigen::ArrayX3i(STACK_SIZE, STACK_COLS);

  std::string curr_op;
  for (int i=0; std::getline(string_stream, curr_op, ','); i++) {
    curr_stack(i/STACK_COLS, i%STACK_COLS) = std::stoi(curr_op);
  }
  indv.command_array = curr_stack;
}


Eigen::ArrayXXd load_agraph_x_vals() {
  std::ifstream filename;
  filename.open(X_FILE);

  Eigen::ArrayXXd x_vals = Eigen::ArrayXXd(NUM_DATA_POINTS, INPUT_DIM);
  std::string curr_x_row;
  for (int row = 0; filename >> curr_x_row; row++) {
    std::stringstream string_stream(curr_x_row);
    std::string curr_x;
    for (int col = 0; std::getline(string_stream, curr_x, ','); col++) {
      x_vals(row, col) = std::stod(curr_x);
    }
  }
  filename.close();
  return x_vals;
}

void run_benchmarks(BenchMarkTestData &benchmark_test_data) {
  Eigen::ArrayXd evaluate_times = time_benchmark(benchmark_evaluate, benchmark_test_data);
  Eigen::ArrayXd x_derivative_times = time_benchmark(benchmark_evaluate_w_x_derivative, benchmark_test_data);
  Eigen::ArrayXd c_derivative_times = time_benchmark(benchmark_evaluate_w_c_derivative, benchmark_test_data);
  print_header();
  print_results(evaluate_times, EVALUATE);
  print_results(x_derivative_times, X_DERIVATIVE);
  print_results(c_derivative_times, C_DERIVATIVE);
}

Eigen::ArrayXd time_benchmark(
  void (*benchmark)(std::vector<AGraphValues>&, Eigen::ArrayXXd&), 
  BenchMarkTestData &test_data, int number, int repeat) {
  Eigen::ArrayXd times = Eigen::ArrayXd(repeat);
  for (int run=0; run<repeat; run++) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<number; i++) {
      benchmark(test_data.indv_list, test_data.x_vals);	
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1>> time_span = (stop - start);
    times(run) = time_span.count();
  }
  return times; 
}

void benchmark_evaluate(std::vector<AGraphValues> &indv_list,
                        Eigen::ArrayXXd &x_vals) {
  std::vector<AGraphValues>::iterator indv;
  for(indv=indv_list.begin(); indv!=indv_list.end(); indv++) {
    SimplifyAndEvaluate(indv->command_array,
                        x_vals,
                        indv->constants);
  } 
}

void benchmark_evaluate_w_x_derivative(std::vector<AGraphValues> &indv_list,
                                       Eigen::ArrayXXd &x_vals) {
  std::vector<AGraphValues>::iterator indv;
  for(indv=indv_list.begin(); indv!=indv_list.end(); indv++) {
    SimplifyAndEvaluateWithDerivative(indv->command_array,
                                      x_vals,
                                      indv->constants,
                                      true);
  }
}

void benchmark_evaluate_w_c_derivative(std::vector<AGraphValues> &indv_list,
                                       Eigen::ArrayXXd &x_vals) {
  std::vector<AGraphValues>::iterator indv;
  for(indv=indv_list.begin(); indv!=indv_list.end(); indv++) {
    SimplifyAndEvaluateWithDerivative(indv->command_array,
                                      x_vals,
                                      indv->constants,
                                      false);
  }
}

void print_header() {
  const std::string top_tacks = std::string(23, '-');
  const std::string title = ":::: PERFORMANCE BENCHMARKS ::::";
  const std::string full_title = top_tacks + title + top_tacks;
  const std::string bottom = std::string (78, '-');
  std::cout << full_title << std::endl;
  output_params("NAME", "MEAN", "STD", " MIN", "MAX");
  std::cout << bottom << std::endl;
}

void print_results(Eigen::ArrayXd &run_times, const std::string &name) {
  double std_dev = standard_deviation(run_times);
  double average = run_times.mean();
  double max = run_times.maxCoeff();
  double min = run_times.minCoeff();
  std::string s_std_dev = string_precision(std_dev, 5);
  std::string s_average= string_precision(average, 5);
  std::string s_min= string_precision(min, 5);
  std::string s_max= string_precision(max, 5);
  output_params(name, s_average, s_std_dev, s_min, s_max);
}

std::string string_precision(double val, int precision) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(precision) << val;
  return stream.str();
}

void output_params(const std::string &name, const std::string &mean, 
                   const std::string &std, const std::string &min, 
                   const std::string &max) {
  std::cout << std::setw(25) << std::left << name << "   "
            << std::setw(10) << std::right << mean << " +- "
            << std::setw(10) << std::left << std << "     "
            << std::setw(10) << std::left << min << "   "
            << std::setw(10) << std::left << max 
            << std::endl;
}

double standard_deviation(Eigen::ArrayXd &vec) {
  return std::sqrt((vec - vec.mean()).square().sum()/(vec.size()-1));
}
