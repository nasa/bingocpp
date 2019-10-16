#ifndef APP_BENCMARK_UTILS_BENCHMARK_DATA_H_
#define APP_BENCMARK_UTILS_BENCHMARK_DATA_H_

#include <BingoCpp/backend.h>
#include <BingoCpp/agraph.h>

using namespace bingo;

struct BenchmarkTestData {
  std::vector<AGraph> indv_list;
  Eigen::ArrayXXd x_vals;
  BenchmarkTestData() {}
  BenchmarkTestData(std::vector<AGraph> &il, Eigen::ArrayXXd &x):
    indv_list(il), x_vals(x) {}
};

void load_benchmark_data(BenchmarkTestData &benchmark_test_data);
void load_agraph_indvidual_data(std::vector<AGraph> &indv_list);
void set_indv_constants(AGraph &indv, std::string &const_string);
void set_indv_stack(AGraph &indv, std::string &stack_string);
Eigen::ArrayXXd load_agraph_x_vals();
double standard_deviation(const Eigen::ArrayXd &vec);

#endif // APP_BENCMARK_UTILS_BENCHMARK_DATA_H_
