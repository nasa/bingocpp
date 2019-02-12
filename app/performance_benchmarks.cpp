#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Core>

#include "BingoCpp/acyclic_graph.h"
#include "BingoCpp/graph_manip.h"

#define STACK_FILE "test-agraph-stacks.csv"
#define CONST_FILE "test-agraph-consts.csv"
#define X_FILE	   "test-agraph-x-vals.csv"
#define INPUT_DIM  4
#define NUM_DATA_POINTS 128 
#define STACK_SIZE  128

struct AGraphValues {
	Eigen::ArrayX3i command_array;
	Eigen::ArrayXXd x_vals;
	Eigen::VectorXd constants;
};

class StatsPrinter {
 public:
	void add_stats(std::string s) {}
	void print() {}
};

void run_benchmarking() {
	StatsPrinter printer = StatsPrinter();
	printer.add_stats("c++: evaluate");
	printer.add_stats("c++: x derivative");
	printer.add_stats("c++: c derivative");

	printer.print();
}

void read_in_stacks(std::vector<Eigen::ArrayX3i> &command_arrays) {
	std::ifstream filename;
	filename.open(STACK_FILE);
	std::string stack_string;
	while(filename >> stack_string){
		std::stringstream string_stream(stack_string);
		std::string curr_op;
		Eigen::ArrayX3i curr_stack(STACK_SIZE, 3);
		for (int i=0; std::getline(string_stream, curr_op, ','); i++) {
			curr_stack(i/3, i%3) = std::stoi(curr_op);
			std::cout<<i<<": "<<curr_op<<std::endl;
		}
		command_arrays.push_back(curr_stack);
	}
}

void read_in_constants(std::vector<Eigen::VectorXd> &constants) {
	std::ifstream filename;
	filename.open(CONST_FILE);
	std::string const_string;

	while(filename >> const_string){
		std::stringstream string_stream(const_string);
		std::string curr_op;
		std::string num_constants;
		std::getline(string_stream, num_constants, ',');
		Eigen::VectorXd curr_const(std::stoi(num_constants));
		for (int i=0; std::getline(string_stream, curr_op, ','); i++) {
			curr_const(i) = std::stod(curr_op);
		}

		constants.push_back(curr_const);
	}
}

void read_in_x_vals(Eigen::ArrayXXd &x_vals) {
	std::ifstream filename;
	filename.open(X_FILE);
	std::string curr_x;
	x_vals = Eigen::ArrayXXd(NUM_DATA_POINTS*INPUT_DIM, INPUT_DIM);
	for (int i=0; std::getline(filename, curr_x, ','); i++) {
		x_vals(i/INPUT_DIM, i%INPUT_DIM) = std::stod(curr_x);
	}
}

void init_a_graphs(std::vector<AGraphValues> &benchmark_test_data) {
	std::vector<Eigen::ArrayX3i> command_arrays; 
	std::vector<Eigen::VectorXd> constants;
	Eigen::ArrayXXd x_vals;
	read_in_stacks(command_arrays);
	read_in_constants(constants);
	read_in_x_vals(x_vals);
}

int main() {
	std::vector<AGraphValues> agraph_vals = std::vector<AGraphValues>();
	init_a_graphs(agraph_vals);	
	run_benchmarking();
	// std::ifstream filename;
	// filename.open("test-agraph-stacks.csv");
	// std::cout<<filename<<std::endl;
	// std::string stack_string;
	// while(filename >> stack_string){
	// 	std::cout<<"stack===========\n";
	// 	std::cout<<stack_string<<std::endl;
	// 	std::stringstream string_stream(stack_string);
	// 	std::string curr_op;
	// 	// Eigen::ArrayX3i curr_stack(STACK_SIZE, 3);
	// 	while(std::getline(string_stream, curr_op, ',')) {
	// 		// curr_stack << std::stoi(curr_op);
	// 		std::cout<<curr_op<<std::endl;
	// 	}
	// 	// command_arrays.push_back(curr_stack);
	// 	// std::cout<<curr_stack<<std::endl;
	// }
	return 0;
}
