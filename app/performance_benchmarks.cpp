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


int main() {
	do_benchmarking();
	return 0;
}

void do_benchmarking() {
	BenchMarkTestData benchmark_test_data = BenchMarkTestData();
	load_benchmark_data(benchmark_test_data);
	run_benchmarks(benchmark_test_data);
}

void load_benchmark_data(BenchMarkTestData &benchmark_test_data) {
	std::vector<AGraphValues> indv_list= std::vector<AGraphValues>();
	load_agraph_indvidual_data(indv_list);
	Eigen::ArrayXXd x_vals = load_agraph_x_vals();

	benchmark_test_data.indv_list = indv_list;
	benchmark_test_data.x_vals = x_vals;
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

	Eigen::ArrayX3i curr_stack = Eigen::ArrayX3i(STACK_SIZE, 3);

	std::string curr_op;
	for (int i=0; std::getline(string_stream, curr_op, ','); i++) {
		curr_stack(i/3, i%3) = std::stoi(curr_op);
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
	print_results(evaluate_times);
	print_results(x_derivative_times);
	print_results(c_derivative_times);
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
		std::chrono::duration<double, std::chrono::seconds> time_span = (stop - start);
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
	std::cout<<"-----------------------"
			<<":::: PERFORMANCE BENCHMARKS ::::"
			<<"-----------------------\n"
			<<"NAME" << std::setw(25) 
			<<std::setw(10) << "MEAN"
			<<std::setw(10) << "STD"
			<<std::setw(10) << "MIN"
			<<std::setw(10) << "MAX" << "\n"
			<< "---------------------------------------------------------\n";
}

void print_results(Eigen::ArrayXd &run_times) {
	double std_dev = standard_deviation(run_times);
	double average = run_times.mean();
	double max = max_val(run_times);
	double min = min_val(run_times);
	std::cout<<std_dev<<std::endl;
	std::cout<<average<<std::endl;
	// std::cout<<max<<std::endl;
	// std::cout<<min<<std::endl;
	
}

double max_val(Eigen::ArrayXd &run_times) {
	double max = run_times(0);
	for (int i=1; i<run_times.rows(); i++) {
		if (run_times(i) > max) 
			max = run_times(i);
	}
	return max;
}

double min_val(Eigen::ArrayXd &run_times) {
	double min = run_times(0);
	for (int i=1; i<run_times.rows(); i++) {
		if (run_times(i) < min) 
			min = run_times(i);
	}
	return min;
}

double standard_deviation(Eigen::ArrayXd &vec) {
	return std::sqrt((vec - vec.mean()).square().sum()/(vec.size()-1));
}
