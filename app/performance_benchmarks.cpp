#include <chrono>

#include <benchmarking/benchmark_data.h>
#include <benchmarking/benchmark_logging.h>

#define EVALUATE "pure c++: evaluate"
#define X_DERIVATIVE "pure c++: x derivative"
#define C_DERIVATIVE "pure c++: c derivative"

void do_benchmarking();
Eigen::ArrayXd time_benchmark(
  void (*benchmark)(const std::vector<AGraph>&, const Eigen::ArrayXXd&), 
  const BenchmarkTestData &test_data, int number=100, int repeat=10);
void run_benchmarks(const BenchmarkTestData &benchmark_test_data);
void benchmark_evaluate(const std::vector<AGraph> &indv_list,
                        const Eigen::ArrayXXd &x_vals);
void benchmark_evaluate_w_x_derivative(const std::vector<AGraph> &indv_list,
                                       const Eigen::ArrayXXd &x_vals);
void benchmark_evaluate_w_c_derivative(const std::vector<AGraph> &indv_list,
                                       const Eigen::ArrayXXd &x_vals);

int main() {
  do_benchmarking();
  return 0;
}

void do_benchmarking() {
  BenchmarkTestData benchmark_test_data =  BenchmarkTestData();
  load_benchmark_data(benchmark_test_data);
  run_benchmarks(benchmark_test_data);
}

void run_benchmarks(const BenchmarkTestData &benchmark_test_data) {
  Eigen::ArrayXd evaluate_times = time_benchmark(benchmark_evaluate, benchmark_test_data);
  Eigen::ArrayXd x_derivative_times = time_benchmark(benchmark_evaluate_w_x_derivative, benchmark_test_data);
  Eigen::ArrayXd c_derivative_times = time_benchmark(benchmark_evaluate_w_c_derivative, benchmark_test_data);
  print_header();
  print_results(evaluate_times, EVALUATE);
  print_results(x_derivative_times, X_DERIVATIVE);
  print_results(c_derivative_times, C_DERIVATIVE);
}

Eigen::ArrayXd time_benchmark(
  void (*benchmark)(const std::vector<AGraph>&, const Eigen::ArrayXXd&), 
  const BenchmarkTestData &test_data, int number, int repeat) {
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

void benchmark_evaluate(const std::vector<AGraph> &indv_list,
                        const Eigen::ArrayXXd &x_vals) {
  std::vector<AGraph>::const_iterator indv;
  for(indv=indv_list.begin(); indv!=indv_list.end(); indv++) {
    bingo::backend::SimplifyAndEvaluate(
      indv->GetCommandArray(), x_vals, indv->GetLocalOptimizationParams());
  } 
}

void benchmark_evaluate_w_x_derivative(const std::vector<AGraph> &indv_list,
                                       const Eigen::ArrayXXd &x_vals) {
  std::vector<AGraph>::const_iterator indv;
  for(indv=indv_list.begin(); indv!=indv_list.end(); indv++) {
    bingo::backend::SimplifyAndEvaluateWithDerivative(
      indv->GetCommandArray(), x_vals, indv->GetLocalOptimizationParams(), true);
  }
}

void benchmark_evaluate_w_c_derivative(const std::vector<AGraph> &indv_list,
                                       const Eigen::ArrayXXd &x_vals) {
  std::vector<AGraph>::const_iterator indv;
  for(indv=indv_list.begin(); indv!=indv_list.end(); indv++) {
    bingo::backend::SimplifyAndEvaluateWithDerivative(
      indv->GetCommandArray(), x_vals, indv->GetLocalOptimizationParams(), false);
  }
}