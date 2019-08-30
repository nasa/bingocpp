#include <limits>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <BingoCpp/agraph.h>
#include <BingoCpp/backend.h>
#include <BingoCpp/constants.h>

const PrintMap kStackPrintMap {
  {2, "({}) + ({})"},
  {3, "({}) - ({})"},
  {4, "({}) * ({})"},
  {5, "({}) / ({}) "},
  {6, "sin ({})"},
  {7, "cos ({})"},
  {8, "exp ({})"},
  {9, "log ({})"},
  {10, "({}) ^ ({})"},
  {11, "abs ({})"},
  {12, "sqrt ({})"},
};

const PrintMap kLatexPrintMap {
  {2, "{} + {}"},
  {3, "{} - ({})"},
  {4, "({})({})"},
  {5, "\\frac{ {} }{ {} }"},
  {6, "sin{ {} }"},
  {7, "cos{ {} }"},
  {8, "exp{ {} }"},
  {9, "log{ {} }"},
  {10, "({})^{ ({}) }"},
  {11, "|{}|"},
  {12, "\\sqrt{ {} }"},
};

const PrintMap kConsolePrintMap {
  {2, "{} + {}"},
  {3, "{} - ({})"},
  {4, "({})({})"},
  {5, "({})/({})"},
  {6, "sin({})"},
  {7, "cos({})"},
  {8, "exp({})"},
  {9, "log({})"},
  {10, "({})^({})"},
  {11, "|{}|"},
  {12, "sqrt({})"},
};

namespace bingo {
namespace {

std::string print_string_with_args(const std::string& string,
                                   const std::string& arg1,
                                   const std::string& arg2) {
  std::stringstream stream;
  bool first_found = false;
  for (auto character = string.begin(); character != string.end(); character++) {
    if (*character == '{' && *(character + 1) == '}') {
      stream << ((!first_found) ? arg1 : arg2);
      character++;
      first_found = true;
    } else {
      stream << *character;
    }
  }
  return stream.str();
}

bool check_optimization_requirement(AGraph& agraph,
                                    const std::vector<bool>& utilized_commands) {
  Eigen::ArrayX3i command_array = agraph.GetCommandArray();
  for (int i = 0; i < command_array.rows(); i++) {
    if (utilized_commands[i] && command_array(i, ArrayProps::kNodeIdx) == Op::LOAD_C) {
      if (command_array(i, ArrayProps::kOp1) == Op::C_OPTIMIZE ||
          command_array(i, ArrayProps::kOp1) >=
          agraph.GetLocalOptimizationParams().size()) {
        return true;
      }
    }
  }
  return false;
}

std::string get_stack_element_string(const AGraph& individual,
                                     int command_index,
                                     const Eigen::ArrayX3i& stack_element) {
  int node = stack_element(0, ArrayProps::kNodeIdx);
  int param1 = stack_element(0, ArrayProps::kOp1);
  int param2 = stack_element(0, ArrayProps::kOp2);

  std::string temp_string = "("+ std::to_string(command_index) +") <= ";
  if (node == Op::LOAD_X) {
    temp_string += "X_" + std::to_string(param1);
  } else if (node == Op::LOAD_C) {
    if (param1 == Op::C_OPTIMIZE ||
        param1 >= individual.GetLocalOptimizationParams().size()) {
      temp_string += "C";
    } else {
      Eigen::VectorXd parameter = individual.GetLocalOptimizationParams();
      temp_string += "C_" + std::to_string(param1) + " = " + 
                     std::to_string(parameter[param1]);
    }
  } else {
    std::string param1_str = std::to_string(param1);
    std::string param2_str = std::to_string(param2);
    temp_string += print_string_with_args(kStackPrintMap.at(node),
                                          param1_str,
                                          param2_str);
  }
  temp_string += '\n';
  return temp_string;
}

std::string get_formatted_element_string(const AGraph& individual,
                                         const Eigen::ArrayX3i& stack_element,
                                         std::vector<std::string> string_list,
                                         const PrintMap& format_map) {
  int node = stack_element(0, ArrayProps::kNodeIdx);
  int param1 = stack_element(0, ArrayProps::kOp1);
  int param2 = stack_element(0, ArrayProps::kOp2);

  std::string temp_string;
  if (node == Op::LOAD_X) {
    temp_string = "X_" + std::to_string(param1);
  } else if (node == Op::LOAD_C) {
    if (param1 == Op::C_OPTIMIZE ||
        param1 >= individual.GetLocalOptimizationParams().size()) {
      temp_string = "?";
    } else {
      Eigen::VectorXd parameter = individual.GetLocalOptimizationParams();
      temp_string = std::to_string(parameter[param1]);
    }
  } else {
    temp_string = print_string_with_args(format_map.at(node),
                                         string_list[param1],
                                         string_list[param2]);
  }
  return temp_string;
}
} // namespace

AGraph::AGraph() {
  command_array_ = Eigen::ArrayX3i(0, 3);
  short_command_array_ = Eigen::ArrayX3i(0, 3);
  constants_ = Eigen::VectorXd(0);
  num_constants_ = 0;
  needs_opt_ = false;
  fitness_ = 1e9;
  fit_set_ = false;
  genetic_age_ = 0;
}

AGraph::AGraph(const AGraph& agraph) {
  this->SetCommandArray(agraph.GetCommandArray());
  constants_ = agraph.GetLocalOptimizationParams();
  needs_opt_ = agraph.NeedsLocalOptimization();
  num_constants_ = agraph.GetNumberLocalOptimizationParams();
  fitness_ = agraph.GetFitness();
  fit_set_ = agraph.IsFitnessSet();
  genetic_age_ = agraph.GetGeneticAge();
}

AGraph AGraph::Copy() {
  AGraph return_val = AGraph(*this);
  return return_val;
}

Eigen::ArrayX3i AGraph::GetCommandArray() const {
  return command_array_;
}

void AGraph::SetCommandArray(Eigen::ArrayX3i command_array) {
  command_array_ = command_array;
  fitness_ = 1e9;
  fit_set_ = false;
  process_modified_command_array();
}

void AGraph::NotifyCommandArrayModificiation() {
  fitness_ = 1e9;
  fit_set_ = false;
  process_modified_command_array();
}

void AGraph::process_modified_command_array() {
  std::vector<bool> util = GetUtilizedCommands();

  needs_opt_ = check_optimization_requirement(*this, util);
  if (needs_opt_)  {
    renumber_constants(util);
  }
  update_short_command_array(util);
}

void AGraph::renumber_constants (const std::vector<bool>& utilized_commands) {
  int const_num = 0;
  int command_array_depth = command_array_.rows();
  for (int i = 0; i < command_array_depth; i++) {
    if (utilized_commands[i] && command_array_(i, ArrayProps::kNodeIdx) == Op::LOAD_C) {
      command_array_.row(i) << 1, const_num , const_num; 
      const_num ++;
    }
  }
  num_constants_ = const_num;
}

void AGraph::update_short_command_array(const std::vector<bool>& utilized_commands) {
  int stack_depth = std::count_if (
      utilized_commands.begin(), utilized_commands.end(), [](bool i) {
        return i;
  });

  short_command_array_ = Eigen::ArrayX3i::Zero(stack_depth, 3);
  int new_index = 0;
  for (size_t old_index = 0; old_index < utilized_commands.size(); old_index++) {
    if (utilized_commands[old_index]) {
      short_command_array_.row(new_index++) = command_array_.row(old_index);
    }
  }

  int inclusive_sum_scan[utilized_commands.size()];
  inclusive_sum_scan[0] = (utilized_commands[0] ? 1 : 0);
  for (size_t i = 1; i < utilized_commands.size(); i++) {
    inclusive_sum_scan[i] = inclusive_sum_scan[i - 1] +
                            (utilized_commands[i] ? 1 : 0);
  }

  for (auto command : short_command_array_.rowwise()) {
    if (!kIsTerminalMap[command(0, ArrayProps::kNodeIdx)]) {
      command(0, ArrayProps::kOp1) = inclusive_sum_scan[command(0, ArrayProps::kOp1)] - 1;
      command(0, ArrayProps::kOp2) = inclusive_sum_scan[command(0, ArrayProps::kOp2)] - 1;
    }
  }
}

double AGraph::GetFitness() const {
  return fitness_;
}

void AGraph::SetFitness(double fitness) {
  fitness_ = fitness;
  fit_set_ = true;
}

bool AGraph::IsFitnessSet() const {
  return fit_set_;
}

void AGraph::SetGeneticAge(const int age) {
  genetic_age_ = age;
}

int AGraph::GetGeneticAge() const {
  return genetic_age_;
}

std::vector<bool> AGraph::GetUtilizedCommands() const {
  return backend::GetUtilizedCommands(command_array_);
}

bool AGraph::NeedsLocalOptimization() const {
  return needs_opt_;
}

int AGraph::GetNumberLocalOptimizationParams() const {
  return num_constants_;
}

void AGraph::SetLocalOptimizationParams(Eigen::VectorXd params) {
  constants_ = params;
  needs_opt_ = false;
}

Eigen::VectorXd AGraph::GetLocalOptimizationParams() const {
  return constants_;
}

Eigen::ArrayXXd AGraph::EvaluateEquationAt(Eigen::ArrayXXd& x) {
  Eigen::ArrayXXd f_of_x; 
  try {
    f_of_x = backend::Evaluate(this->command_array_,
                               x,
                               this->constants_);
    return f_of_x;
  } catch (const std::underflow_error& ue) {
    return Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
  } catch (const std::overflow_error& oe) {
    return Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
  } 
}

EvalAndDerivative AGraph::EvaluateEquationWithXGradientAt(Eigen::ArrayXXd& x) {
  EvalAndDerivative df_dx;
  try {
    df_dx = backend::EvaluateWithDerivative(this->command_array_,
                                            x,
                                            this->constants_,
                                            true);
    return df_dx;
  } catch (const std::underflow_error& ue) {
    Eigen::ArrayXXd nan_array = Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    return std::make_pair(nan_array, nan_array);
  } catch (const std::overflow_error& oe) {
    Eigen::ArrayXXd nan_array = Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    return std::make_pair(nan_array, nan_array);
  }
}

EvalAndDerivative AGraph::EvaluateEquationWithLocalOptGradientAt(
    Eigen::ArrayXXd& x) {
  EvalAndDerivative df_dc;
  try {
    df_dc = backend::EvaluateWithDerivative(this->command_array_,
                                            x,
                                            this->constants_,
                                            false);
    return df_dc;
  } catch (const std::underflow_error& ue) {
    Eigen::ArrayXXd nan_array = Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    return std::make_pair(nan_array, nan_array);
  } catch (const std::overflow_error& oe) {
    Eigen::ArrayXXd nan_array = Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    return std::make_pair(nan_array, nan_array);
  }
}

std::ostream& operator<<(std::ostream& strm, const AGraph& graph) {
  return strm << graph.GetConsoleString();
}

std::string AGraph::GetLatexString() const {
  return get_formatted_string_using(kLatexPrintMap);
}

std::string AGraph::GetConsoleString() const {
  return get_formatted_string_using(kConsolePrintMap);
}

std::string AGraph::get_formatted_string_using(const PrintMap& format_map) const {
  std::vector<std::string> string_list;
  for (auto stack_element : short_command_array_.rowwise()) {
    std::string temp_string = get_formatted_element_string(
        *this, stack_element, string_list, format_map);
    string_list.push_back(temp_string);
  }
  return string_list.back();
}

std::string AGraph::GetStackString() const {
  std::stringstream print_str; 
  print_str << "---full stack---\n"
            << get_stack_string()
            << "---small stack---\n"
            << get_stack_string(true); 
  return print_str.str();
}

std::string AGraph::get_stack_string(const bool is_short) const {
  Eigen::ArrayX3i stack;
  if (is_short) {
    stack = short_command_array_; 
  } else {
    stack = command_array_;
  }
  std::string temp_string;
  for (int i = 0; i < stack.rows(); i++) {
    temp_string += get_stack_element_string(*this, i, stack.row(i));
  }
  return temp_string;
}

int AGraph::GetComplexity() const {
  std::vector<bool> commands = GetUtilizedCommands();
  return std::count_if (commands.begin(), commands.end(), [](bool i) {
    return i;
  });
}

bool AGraph::HasArityTwo(int node) {
  return kIsArity2Map[node];
}

bool AGraph::IsTerminal(int node) {
  return kIsTerminalMap[node];
}

const bool AGraph::kIsArity2Map[13] = {
  false,
  false,
  true,
  true,
  true,
  true,
  false,
  false,
  false,
  false,
  true,
  false,
  false
};

const bool AGraph::kIsTerminalMap[13] = {
  true,
  true,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false
};
}