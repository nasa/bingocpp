#ifndef APP_BENCMARK_UTILS_BENCHMARK_LOGGING_H_
#define APP_BENCMARK_UTILS_BENCHMARK_LOGGING_H_

#include <string>

#include <Eigen/Core>

void print_header(std::string title="PERFORMANCE BENCHMARKS");
void print_results(const Eigen::ArrayXd &run_times, const std::string &name);
std::string string_precision(double val, int precision);
void output_params(const std::string &name, const std::string &mean, 
                   const std::string &std, const std::string &min, 
                   const std::string &max); 

#endif //APP_BENCHMARK_UTILS_BENCHMARK_LOGGING_H_