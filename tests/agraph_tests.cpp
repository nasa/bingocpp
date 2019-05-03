#include <unordered_map>
#include <string>
#include <climits>

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <BingoCpp/agraph.h>
#include <BingoCpp/backend.h>

#include "test_fixtures.h"
#include "testing_utils.h"

using namespace bingo;

typedef std::unordered_map<std::string, std::string> GraphString;
typedef std::unordered_map<std::string , GraphString> StringMap;

namespace {

struct AGraphTestVals {
  Eigen::ArrayXXd x;
  Eigen::ArrayXXd f_of_x;
  Eigen::ArrayXXd grad_x;
  Eigen::ArrayXXd grad_c;

  AGraphTestVals() {}
  AGraphTestVals(Eigen::ArrayXXd x_,
                 Eigen::ArrayXXd f_of_x_,
                 Eigen::ArrayXXd grad_x_,
                 Eigen::ArrayXXd grad_c_) : 
      x(x_),
      f_of_x(f_of_x_),
      grad_x(grad_x_),
      grad_c(grad_c_) {}
};

class AGraphTest : public ::testing::TestWithParam<std::string> {
 public:
  AGraph invalid_graph;
  AGraph sample_agraph_1;
  AGraph all_funcs_graph;
  StringMap map_to_graph_string;
  std::unordered_map<std::string, AGraph> map_to_graph;
  AGraphTestVals sample_agraph_1_values;

  void SetUp() {
    sample_agraph_1 = testutils::init_sample_agraph_1();
    invalid_graph = init_invalid_graph(sample_agraph_1);
    all_funcs_graph = init_all_funcs_graph();
    map_to_graph_string = init_hash_map();
    map_to_graph = {
      {"all_funcs_graph", all_funcs_graph},
      {"sample_agraph_1", sample_agraph_1},
      {"invalid_graph", invalid_graph}
    };
    sample_agraph_1_values = init_sample_agraph_1_values();
  }

  void TearDown() {}

  AGraph init_invalid_graph(AGraph& agraph) {
    AGraph return_val = AGraph();
    return_val.setCommandArray(agraph.getCommandArray());
    return return_val;
  }

  AGraphTestVals init_sample_agraph_1_values() {
    int num_points = 11;
    Eigen::ArrayXXd x(num_points, 2);
    x.col(0) = Eigen::ArrayXd::LinSpaced(num_points, -1, 0);
    x.col(1) = Eigen::ArrayXd::LinSpaced(num_points, 0, 1);
    Eigen::ArrayXXd f_of_x = (x + 1.0).sin() + 1.0;
    Eigen::ArrayXXd grad_x = Eigen::ArrayXXd::Zero(x.rows(), x.cols());
    grad_x.col(0) = (x.col(0) + 1.0).cos();
    Eigen::ArrayXXd grad_c = (x + 1.0).cos() + 1.0;
    return AGraphTestVals(x, f_of_x.col(0), grad_x, grad_c.col(0));
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
                         "0))(X_0))/(X_0))))))^(X_0)|)"},
      {"complexity", "13"}
    };
    GraphString sample_agraph_1_map {
      {"latex string", "sin{ X_0 + 1.000000 } + 1.000000"},
      {"console string", "sin(X_0 + 1.000000) + 1.000000"},
      {"complexity", "5"}
    };
    GraphString invalid_graph_map {
      {"latex string", "sin{ X_0 + ? } + ?"},
      {"console string", "sin(X_0 + ?) + ?"},
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
  ASSERT_DOUBLE_EQ(agraph_copy.getLocalOptimizationParams()[0], 1.0);
}

TEST_P(AGraphTest, latex_print) {
  std::string agraph_name = GetParam();
  std::string string_rep = map_to_graph_string.at(agraph_name).at("latex string");
  AGraph agraph = map_to_graph.at(agraph_name);
  ASSERT_STREQ(string_rep.c_str(), agraph.getLatexString().c_str());
}

TEST_P(AGraphTest, console_print) {
  std::string agraph_name = GetParam();
  std::string string_rep = map_to_graph_string.at(agraph_name).at("console string");
  AGraph agraph = map_to_graph.at(agraph_name);
  std::stringstream ss;
  ss << agraph;
  ASSERT_STREQ(string_rep.c_str(), ss.str().c_str());
}

TEST_P(AGraphTest, complexity_print) {
  std::string agraph_name = GetParam();
  int complexity_val = std::stoi(map_to_graph_string.at(agraph_name).at("complexity"));
  AGraph agraph = map_to_graph.at(agraph_name);
  ASSERT_EQ(complexity_val, agraph.getComplexity());
}

INSTANTIATE_TEST_CASE_P(,AGraphTest, ::testing::Values(
    "all_funcs_graph", "sample_agraph_1", "invalid_graph"));

TEST_F(AGraphTest, stack_print) {
  std::stringstream expected_str;
  expected_str << "---full stack---\n"
                   "(0) <= X_0\n"
                   "(1) <= C_0 = 1.000000\n"
                   "(2) <= (0) + (1)\n"
                   "(3) <= sin (2)\n" 
                   "(4) <= (0) + (1)\n"
                   "(5) <= (3) + (1)\n"
                   "---small stack---\n"
                   "(0) <= X_0\n"
                   "(1) <= C_0 = 1.000000\n"
                   "(2) <= (0) + (1)\n"
                   "(3) <= sin (2)\n"
                   "(4) <= (3) + (1)\n";
  ASSERT_STREQ(expected_str.str().c_str(), sample_agraph_1.getStackString().c_str());
}

TEST_F(AGraphTest, invalid_stack_print) {
  std::stringstream expected_str;
  expected_str << "---full stack---\n"
                   "(0) <= X_0\n"
                   "(1) <= C\n"
                   "(2) <= (0) + (1)\n"
                   "(3) <= sin (2)\n" 
                   "(4) <= (0) + (1)\n"
                   "(5) <= (3) + (1)\n"
                   "---small stack---\n"
                   "(0) <= X_0\n"
                   "(1) <= C\n"
                   "(2) <= (0) + (1)\n"
                   "(3) <= sin (2)\n"
                   "(4) <= (3) + (1)\n";
  ASSERT_STREQ(expected_str.str().c_str(), invalid_graph.getStackString().c_str());
}

TEST_F(AGraphTest, evaluateAt) {
  ASSERT_TRUE(testutils::almost_equal(
    sample_agraph_1_values.f_of_x,
    sample_agraph_1.evaluateEquationAt(sample_agraph_1_values.x)
  ));
}

TEST_F(AGraphTest, evaluatWithXDerivative) {
  Eigen::ArrayXXd x = sample_agraph_1_values.x;
  EvalAndDerivative result = sample_agraph_1.evaluateEquationWithXGradientAt(x);
  Eigen::ArrayXXd f_of_x = result.first;
  Eigen::ArrayXXd df_dx = result.second;
  ASSERT_TRUE(testutils::almost_equal(sample_agraph_1_values.f_of_x, f_of_x));
  ASSERT_TRUE(testutils::almost_equal(sample_agraph_1_values.grad_x, df_dx));
}

TEST_F(AGraphTest, evaluatWithCDerivative) {
  Eigen::ArrayXXd x = sample_agraph_1_values.x;
  EvalAndDerivative result = sample_agraph_1.evaluateEquationWithLocalOptGradientAt(x);
  Eigen::ArrayXXd f_of_x = result.first;
  Eigen::ArrayXXd df_dc = result.second;
  ASSERT_TRUE(testutils::almost_equal(sample_agraph_1_values.f_of_x, f_of_x));
  ASSERT_TRUE(testutils::almost_equal(sample_agraph_1_values.grad_c, df_dc));
}

TEST_F(AGraphTest, get_number_of_optimization_params) {
  ASSERT_TRUE(invalid_graph.needsLocalOptimization());
}

TEST_F(AGraphTest, getNumberOfOptimizationParams) {
  ASSERT_EQ(invalid_graph.getNumberLocalOptimizationParams(), 1);
}

TEST_F(AGraphTest, setOptimizationParams) {
  Eigen::VectorXd opt_params(1);
  opt_params << 1.0;
  invalid_graph.setLocalOptimizationParams(opt_params);
  ASSERT_FALSE(invalid_graph.needsLocalOptimization());
  ASSERT_TRUE(testutils::almost_equal(
    invalid_graph.evaluateEquationAt(sample_agraph_1_values.x),
    sample_agraph_1.evaluateEquationAt(sample_agraph_1_values.x))
  );
}

TEST_F(AGraphTest, setting_fitness_updates_fit_set) {
  AGraph new_graph = AGraph();
  ASSERT_FALSE(new_graph.isFitnessSet());
  new_graph.setFitness(2.0);
  ASSERT_TRUE(new_graph.isFitnessSet());
}

TEST_F(AGraphTest, notify_command_array_mod) {
  ASSERT_TRUE(sample_agraph_1.isFitnessSet());
  sample_agraph_1.notifyCommandArrayModificiation();
  ASSERT_FALSE(sample_agraph_1.isFitnessSet());
}

TEST_F(AGraphTest, setting_command_array_unsets_fitness) {
  ASSERT_TRUE(sample_agraph_1.isFitnessSet());
  sample_agraph_1.setCommandArray(Eigen::ArrayX3i::Ones(1, 3));
  ASSERT_FALSE(sample_agraph_1.isFitnessSet());
}

// class AGraphExceptionTest : public ::testing::Test {
//  public:
//   AGraph x_squared;
//   Eigen::ArrayXXd large_x;

//   void SetUp() {
//     x_squared = init_exception_graph();
//     large_x = init_large_x_vals();
//   }

//   void TearDown() {}

//   AGraph init_exception_graph() {
//     Eigen::ArrayX3i command_array(2, 3);
//     command_array << 0, 0, 0,
//                      4, 0, 0;
//     AGraph test_stack = AGraph();
//     test_stack.setCommandArray(command_array);
//     return test_stack;
//   }

//   Eigen::ArrayXXd init_large_x_vals() {
//     int num_points = 100;
//     double begin = 2;
//     double end = 12;
//     Eigen::ArrayXXd result(num_points, 1);
//     result = Eigen::ArrayXd::LinSpaced(num_points, begin, end);
//     return result;
//   }
// };

// TEST_F(AGraphExceptionTest, evaluate_overflow_returns_nan_array) {
//   std::cout << large_x << std::endl;
//   Eigen::ArrayXXd return_vals = x_squared.evaluateEquationAt(large_x);
// }
} // namespace