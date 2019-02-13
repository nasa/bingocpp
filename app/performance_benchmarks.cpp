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

class StatsPrinter {
 public:
	void add_stats(std::string s) {}
	void print() {}
};

void benchmark_evaluate(std::vector<AGraphValues> &agraph_vals,
												Eigen::ArrayXXd &x_vals) {
	std::vector<AGraphValues>::iterator indv;
	for(indv=agraph_vals.begin(); indv!=agraph_vals.end(); indv++) {
		SimplifyAndEvaluate(indv->command_array,
												x_vals,
												indv->constants);
	} 
}

void benchmark_evaluate_w_x_derivative(std::vector<AGraphValues> &agraph_vals,
																			 Eigen::ArrayXXd &x_vals) {
	std::vector<AGraphValues>::iterator indv;
	for(indv=agraph_vals.begin(); indv!=agraph_vals.end(); indv++) {
		SimplifyAndEvaluateWithDerivative(indv->command_array,
																			x_vals,
																			indv->constants,
																			true);
	}
}

void benchmark_evaluate_w_c_derivative(std::vector<AGraphValues> &agraph_vals,
																			 Eigen::ArrayXXd &x_vals) {
	std::vector<AGraphValues>::iterator indv;
	for(indv=agraph_vals.begin(); indv!=agraph_vals.end(); indv++) {
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
	// std::cout<<"d"<<std::endl;
	Eigen::ArrayX3i curr_stack = Eigen::ArrayX3i(STACK_SIZE, 3);
// std::cout<<"e"<<std::endl;
	std::string curr_op;
	for (int i=0; std::getline(string_stream, curr_op, ','); i++) {
		curr_stack(i/3, i%3) = std::stoi(curr_op);
	}
	// std::cout<<"e"<<std::endl;
	indv.command_array = curr_stack;
}

void set_indv_constants(AGraphValues &indv, std::string &const_string) {
	std::stringstream string_stream(const_string);

	std::string num_constants;
	std::getline(string_stream, num_constants, ',');
	Eigen::VectorXd curr_const(std::stoi(num_constants));

	std::string curr_op;
	for (int i=0; std::getline(string_stream, curr_op, ','); i++) {
		curr_const(i) = std::stod(curr_op);
	}
	indv.constants = curr_const;
}

Eigen::ArrayXXd load_agraph_x_vals() {
	Eigen::ArrayXXd x_vals = Eigen::ArrayXXd(NUM_DATA_POINTS, INPUT_DIM);

	std::ifstream filename;
	filename.open(X_FILE);

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

void load_agraph_indvidual_data(std::vector<AGraphValues> &agraph_vals,
																const char *filename) {
	std::ifstream filestream;
	filestream.open(filename);

	std::string file_line;
	for (int indv=0; filestream >> file_line; indv++) {
		// std::cout<<"a"<<std::endl;
		AGraphValues curr_indv = AGraphValues();
		// if (strcmp(filename, STACK_FILE)) 
		set_indv_stack(curr_indv, file_line);
		// else if (strcmp(filename, CONST_FILE)) 
		// 	set_indv_constants(curr_indv, file_line);
		// std::cout<<"b"<<std::endl;
		agraph_vals.push_back(curr_indv);
		// std::cout<<"c"<<std::endl;
	}
	filestream.close();
}

void init_a_graphs(std::vector<AGraphValues> &benchmark_test_data) {
	// load_agraph_indvidual_data(benchmark_test_data, STACK_FILE);
	// load_agraph_indvidual_data(benchmark_test_data, CONST_FILE);
	// std::cout<<"1"<<std::endl;
	Eigen::ArrayXXd x_vals = load_agraph_x_vals();
// std::cout<<"2"<<std::endl;
	// std::cout<<benchmark_test_data[0].command_array<<std::endl;
// std::cout<<"3"<<std::endl;
	// std::cout<<benchmark_test_data[1].constants<<std::endl;
	std::cout<<x_vals<<std::endl;
// std::cout<<"4"<<std::endl;
}

int main() {
	std::vector<AGraphValues> agraph_vals = std::vector<AGraphValues>();
	init_a_graphs(agraph_vals);	
	run_benchmarking();
	return 0;
}
