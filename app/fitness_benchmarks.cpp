#include <chrono>

#include <BingoCpp/agraph.h>
#include <BingoCpp/explicit_regression.h>
#include <BingoCpp/implicit_regression.h>
#include <BingoCpp/utils.h>

#include "benchmarking/benchmark_data.h"
#include "benchmarking/benchmark_logging.h"

#define EXPLICIT "explicit regression"
#define IMPLICIT "implicit regression"

using namespace bingo;

void benchmark_regression(const std::vector<AGraph> &agraph_list,
                          const VectorBasedFunction &fitness_function);
Eigen::ArrayXd time_benchmark(
    void (*benchmark)(const std::vector<AGraph>&, const VectorBasedFunction &), 
    const BenchmarkTestData &test_data, 
    const VectorBasedFunction &fitness_function, int number=100, int repeat=10);
void do_regression_benchmarking();
void run_regression_benchmarks(const BenchmarkTestData &benchmark_test_data);

int main() {
  do_regression_benchmarking();
  return 0;
}

void do_regression_benchmarking() {
  BenchmarkTestData benchmark_test_data;
  load_benchmark_data(benchmark_test_data);
  run_regression_benchmarks(benchmark_test_data);
}

void run_regression_benchmarks(const BenchmarkTestData &benchmark_test_data) {
  auto input_and_derivative = CalculatePartials(benchmark_test_data.x_vals);
  auto x_vals = input_and_derivative.first;
  auto derivative = input_and_derivative.second;
  auto y = Eigen::ArrayXXd::Zero(x_vals.rows(), x_vals.cols());
  auto e_training_data = new ExplicitTrainingData(x_vals, y);
  ExplicitRegression e_regression(e_training_data);
  Eigen::ArrayXd explicit_times = time_benchmark(benchmark_regression,
                                                 benchmark_test_data,
                                                 e_regression);
  auto i_training_data = new ImplicitTrainingData(x_vals, derivative);
  ImplicitRegression i_regression(i_training_data);
  Eigen::ArrayXd implicit_times = time_benchmark(benchmark_regression,
                                                 benchmark_test_data,
                                                 e_regression);
  print_header();
  print_results(explicit_times, EXPLICIT);
  print_results(implicit_times, IMPLICIT);
  delete i_training_data;
  delete e_training_data;
}

Eigen::ArrayXd time_benchmark(
  void (*benchmark)(const std::vector<AGraph>&, const VectorBasedFunction &), 
  const BenchmarkTestData &test_data, 
  const VectorBasedFunction &fitness_function, int number, int repeat) {
  Eigen::ArrayXd times = Eigen::ArrayXd(repeat);
  for (int run=0; run<repeat; run++) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<number; i++) {
      benchmark(test_data.indv_list, fitness_function);	
    }
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1, 1>> time_span = (stop - start);
    times(run) = time_span.count();
  }
  return times; 
}

void benchmark_regression(const std::vector<AGraph> &agraph_list,
                          const VectorBasedFunction &fitness_function) {
  std::vector<AGraph>::const_iterator indv;
  for(indv = agraph_list.begin(); indv != agraph_list.end(); indv ++) {
    fitness_function.EvaluateIndividualFitness(*indv);
  }
}