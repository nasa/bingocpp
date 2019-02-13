#include <iostream>
#include <fstream>
#include <sstream>
#include <istream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <Eigen/Core>

#include "BingoCpp/acyclic_graph.h"

#define STACK_FILE			"test-agraph-stacks.csv"
#define CONST_FILE			"test-agraph-consts.csv"
#define X_FILE					"test-agraph-x-vals.csv"
#define INPUT_DIM				4
#define NUM_DATA_POINTS 128 
#define STACK_SIZE  		128
#define STACK_COLS			3

struct AGraphValues {
	Eigen::ArrayX3i command_array;
	Eigen::VectorXd constants;
};

struct BenchMarkTestData {
	std::vector<AGraphValues> indv_list;
	Eigen::ArrayXXd x_vals;
};

class StatsPrinter {
 public:
	void add_stats(std::string s) {}
	void print() {}
};

Eigen::ArrayXd time_benchmark(
	std::function<void(std::vector<AGraphValues>, Eigen::ArrayXXd)> benchmark, 
	BenchMarkTestData &test_data, int number=100, int repeat=10) {
	Eigen::ArrayXd times = Eigen::ArrayXd(repeat);
	for (int run=0; run<repeat; run++) {
		auto start = std::chrono::high_resolution_clock::now();
		for (int i=0; i<number; i++) {
			benchmark(test_data.indv_list, test_data.x_vals);	
		}
		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> time_span = (start - stop);
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

void run_benchmarking() {
	StatsPrinter printer = StatsPrinter();
	printer.add_stats("c++: evaluate");
	printer.add_stats("c++: x derivative");
	printer.add_stats("c++: c derivative");

	printer.print();
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

void set_indv_constants(AGraphValues &indv, std::string &const_string) {
	std::stringstream string_stream(const_string);

	std::string num_constants;
	std::getline(string_stream, num_constants, ',');
	Eigen::VectorXd curr_const(std::stoi(num_constants));

	std::string curr_val;
	for (int i=0; std::getline(string_stream, curr_val, ','); i++) {
		curr_const(i) = std::stod(curr_val);
	}
	indv.constants = curr_const;
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

void load_agraph_indvidual_data(std::vector<AGraphValues> &indv_list) {
	std::ifstream stack_filestream;
	std::ifstream const_filestream;
	stack_filestream.open(STACK_FILE);
	const_filestream.open(CONST_FILE);

	std::string stack_file_line;
	std::string const_file_line;
	for (int indv=0; 
			(stack_filestream >> stack_file_line) && 
			(const_filestream >> const_file_line); 
			indv++) {
		AGraphValues curr_indv = AGraphValues();
		set_indv_stack(curr_indv, stack_file_line);
		set_indv_constants(curr_indv, const_file_line);
		indv_list.push_back(curr_indv);
	}
	stack_filestream.close();
	const_filestream.close();
}

void load_benchmark_data(BenchMarkTestData &benchmark_test_data) {
	std::vector<AGraphValues> indv_list= std::vector<AGraphValues>();

	load_agraph_indvidual_data(indv_list);
	load_agraph_indvidual_data(indv_list);
	Eigen::ArrayXXd x_vals = load_agraph_x_vals();

	benchmark_test_data.indv_list = indv_list;
	benchmark_test_data.x_vals = x_vals;
	// std::cout<<benchmark_test_data[0].command_array<<std::endl;
	// std::cout<<benchmark_test_data[1].constants<<std::endl;
	// std::cout<<x_vals<<std::endl;

}

int main() {
	BenchMarkTestData benchmark_test_data = BenchMarkTestData();
	load_benchmark_data(benchmark_test_data);	
	run_benchmarking();
	return 0;
}
