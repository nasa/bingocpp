#include <unordered_map>
#include <string>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <BingoCpp/agraph.h>
#include <BingoCpp/backend.h>

using namespace bingo;

typedef std::unordered_map<std::string, std::string> GraphString;
typedef std::unordered_map<std::string , GraphString> StringMap;

namespace {

class AGraphTest : public ::testing::TestWithParam<std::string> {
 public:
  AGraph invalid_graph;
  AGraph sample_agraph_1;
  AGraph all_funcs_graph;
  StringMap map_to_graph_string;
  std::unordered_map<std::string, AGraph> map_to_graph;

  void SetUp() {
    sample_agraph_1 = init_sample_agraph_1();
    invalid_graph = init_invalid_graph(sample_agraph_1);
    all_funcs_graph = init_all_funcs_graph();
    map_to_graph_string = init_hash_map();
    map_to_graph = {
      {"all_funcs_graph", all_funcs_graph},
      {"sample_agraph_1", sample_agraph_1},
      {"invalid_graph", invalid_graph}
    };
  }

  void TearDown() {

  }

  AGraph init_invalid_graph(AGraph& agraph) {
    AGraph return_val = AGraph();
    return_val.setCommandArray(agraph.getCommandArray());
    return return_val;
  }
  
  AGraph init_sample_agraph_1() {
    AGraph test_graph = AGraph();
    Eigen::ArrayX3i test_command_array(6, 3);
    test_command_array << 0, 0, 0,
                          1, 0, 0,
                          2, 0, 1,
                          6, 2, 2,
                          2, 0, 1,
                          2, 3, 1;
    test_graph.setCommandArray(test_command_array);
    test_graph.setGeneticAge(10);
    Eigen::VectorXd local_opt_params(2);
    local_opt_params << 1.0, 1.0;
    test_graph.setLocalOptimizationParams(local_opt_params);
    test_graph.setFitness(1);
    return test_graph;
  }

  AGraph init_sample_agraph_2() {
    AGraph test_graph = AGraph();
    Eigen::ArrayX3i test_command_array(6, 3);
    test_command_array << 0, 1, 3,
                          1, 1, 2,
                          3, 1, 1,
                          4, 0, 2,
                          2, 0, 1,
                          6, 3, 0;
    test_graph.setCommandArray(test_command_array);
    test_graph.setGeneticAge(20);
    Eigen::VectorXd local_opt_params(2);
    local_opt_params << 1.0, 1.0;
    test_graph.setLocalOptimizationParams(local_opt_params);
    test_graph.setFitness(2);
    return test_graph;
  }

  AGraph init_all_funcs_graph() {
    AGraph test_graph = AGraph();
    test_graph.setGeneticAge(10);
    Eigen::ArrayX3i command_array(13, 3);
    command_array << 0, 0, 0,
                     1, 0, 0,
                     2, 1, 0,
                     3, 2, 0,
                     4, 3, 0,
                     5, 4, 0,
                     6, 5, 0,
                     7, 6, 0,
                     8, 7, 0,
                     9, 8, 0,
                     10, 9, 0,
                     11, 10, 0,
                     12, 11, 0;
    test_graph.setCommandArray(command_array);
    Eigen::VectorXd local_opt_params(2);
    local_opt_params << 1.0, 1.0;
    test_graph.setLocalOptimizationParams(local_opt_params);
    return test_graph;
  }

  StringMap init_hash_map() {
    StringMap return_val;
    GraphString all_funcs_graph_map {
      {"latex string", "\\sqrt{ |(log{ exp{ cos{ sin{ \\frac{ (1.000000"
                       " + X_0 - (X_0))(X_0) }{ X_0 } } } } })^{ ("
                       "X_0) }| }"},
      {"console string", "sqrt(|(log(exp(cos(sin(((1.000000 + X_0 - (X_"
                         "0))(X_0))/(X_0) )))))^(X_0)|)"},
      {"complexity", "13"}
    };
    GraphString sample_agraph_1_map {
      {"latex string", "sin{ X_0 + 1.000000 } + 1.000000"},
      {"console string", "sin{ X_0 + 1.000000 } + 1.000000"},
      {"complexity", "5"}
    };
    GraphString invalid_graph_map {
      {"latex string", "sin{ X_0 + ? } + ?"},
      {"console string", "sin{ X_0 + ? } + ?"},
      {"complexity", "5"}
    };
    return_val.insert({"all_funcs_graph", all_funcs_graph_map});
    return_val.insert({"sample_agraph_1", sample_agraph_1_map});
    return_val.insert({"invalid_graph", invalid_graph_map});
    return return_val;
  }
};

TEST_F(AGraphTest, copy) {
  AGraph agraph_copy = sample_agraph_1.copy();

  Eigen::ArrayX3i command_array = sample_agraph_1.getCommandArray();
  command_array(1, 1) = 100;
  sample_agraph_1.setCommandArray(command_array);
  Eigen::VectorXd constants = sample_agraph_1.getLocalOptimizationParams();
  constants[0] = 100;
  sample_agraph_1.setLocalOptimizationParams(constants);

  ASSERT_EQ(agraph_copy.getGeneticAge(), 10);
  ASSERT_EQ(agraph_copy.getCommandArray()(1,1), 0);
  ASSERT_DOUBLE_EQ(agraph_copy.getLocalOptimizationParams()(0), 1.0);
}

TEST_P(AGraphTest, latex_print) {
  std::string agraph_name = GetParam();
  std::string string_rep = map_to_graph_string.at(agraph_name).at("latex string");
  AGraph agraph = map_to_graph.at(agraph_name);
  ASSERT_STREQ(string_rep.c_str(), agraph.getLatexString().c_str());
}
INSTANTIATE_TEST_CASE_P(,AGraphTest, ::testing::Values(
    "all_funcs_graph", "sample_agraph_1", "invalid_graph"));
} // namespace