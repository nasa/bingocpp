#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "BingoCpp/agraph.h"
#include "BingoCpp/agraph_maps.h"
#include "BingoCpp/backend.h"
#include "BingoCpp/constants.h"

namespace bingo {
namespace {

const double kFitnessNotSet = 1e9;

bool check_optimization_requirement(
    AGraph &agraph,
    const std::vector<bool> &utilized_commands);

int renumber_constants(const std::vector<bool> &utilized_commands,
                       Eigen::ArrayX3i &command_array);

std::string get_formatted_string_using(
    const PrintMap &format_map,
    const AGraph &agraph,
    const Eigen::ArrayX3i &short_command_array);

std::string get_formatted_element_string(const AGraph &individual,
                                         const Eigen::ArrayX3i &stack_element,
                                         std::vector<std::string> string_list,
                                         const PrintMap &format_map);

std::string print_string_with_args(const std::string &string,
                                   const std::string &arg1,
                                   const std::string &arg2);

std::string get_stack_string(const AGraph &agraph,
                             const Eigen::ArrayX3i &command_array);

std::string get_stack_element_string(const AGraph &individual,
                                     int command_index,
                                     const Eigen::ArrayX3i &stack_element);
} // namespace

AGraph::AGraph(bool manual_constants) {
  command_array_ = Eigen::ArrayX3i(0, 3);
  short_command_array_ = Eigen::ArrayX3i(0, 3);
  constants_ = Eigen::VectorXd(0);
  needs_opt_ = false;
  num_constants_ = 0;
  manual_constants_ = manual_constants;
  fitness_ = kFitnessNotSet;
  fit_set_ = false;
  genetic_age_ = 0;
}
  
AGraph::AGraph(const AGraph &agraph) {
  command_array_ = agraph.command_array_;
  short_command_array_ = agraph.short_command_array_;
  constants_ = agraph.constants_;
  needs_opt_ = agraph.needs_opt_;
  num_constants_ = agraph.num_constants_;
  manual_constants_ = agraph.manual_constants_;
  fitness_ = agraph.fitness_;
  fit_set_ = agraph.fit_set_;
  genetic_age_ = agraph.genetic_age_;
}

AGraph AGraph::Copy() {
  return AGraph(*this);
}

const Eigen::ArrayX3i &AGraph::GetCommandArray() const {
  return command_array_;
}

Eigen::ArrayX3i &AGraph::GetCommandArrayModifiable() {
  return command_array_;
}

void AGraph::SetCommandArray(const Eigen::ArrayX3i &command_array) {
  command_array_ = command_array;
  NotifyCommandArrayModificiation();
}

void AGraph::NotifyCommandArrayModificiation() {
  fitness_ = kFitnessNotSet;
  fit_set_ = false;
  process_modified_command_array();
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

void AGraph::SetFitnessStatus(bool val) {
  fit_set_ = val;
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

const Eigen::VectorXd &AGraph::GetLocalOptimizationParams() const {
  return constants_;
}

Eigen::VectorXd &AGraph::GetLocalOptimizationParamsModifiable() {
  return constants_;
}

Eigen::ArrayXXd 
AGraph::EvaluateEquationAt(const Eigen::ArrayXXd &x) const {
  Eigen::ArrayXXd f_of_x; 
  try {
    f_of_x = backend::Evaluate(this->short_command_array_,
                               x,
                               this->constants_);
    return f_of_x;
  } catch (const std::underflow_error &ue) {
    return Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
  } catch (const std::overflow_error &oe) {
    return Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
  } 
}

EvalAndDerivative
AGraph::EvaluateEquationWithXGradientAt(const Eigen::ArrayXXd &x) const {
  EvalAndDerivative df_dx;
  try {
    df_dx = backend::EvaluateWithDerivative(this->short_command_array_,
                                            x,
                                            this->constants_,
                                            true);
    return df_dx;
  } catch (const std::underflow_error &ue) {
    Eigen::ArrayXXd nan_array =
        Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    return std::make_pair(nan_array, nan_array);
  } catch (const std::overflow_error &oe) {
    Eigen::ArrayXXd nan_array =
        Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    return std::make_pair(nan_array, nan_array);
  }
}

EvalAndDerivative
AGraph::EvaluateEquationWithLocalOptGradientAt(const Eigen::ArrayXXd &x) const {
  EvalAndDerivative df_dc;
  try {
    df_dc = backend::EvaluateWithDerivative(this->short_command_array_,
                                            x,
                                            this->constants_,
                                            false);
    return df_dc;
  } catch (const std::underflow_error &ue) {
    Eigen::ArrayXXd nan_array =
        Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    return std::make_pair(nan_array, nan_array);
  } catch (const std::overflow_error &oe) {
    Eigen::ArrayXXd nan_array =
        Eigen::ArrayXXd::Constant(x.rows(), x.cols(), kNaN);
    return std::make_pair(nan_array, nan_array);
  }
}

std::ostream &operator<<(std::ostream &strm, const AGraph &graph) {
  return strm << graph.GetConsoleString();
}

std::string AGraph::GetLatexString() const {
  return get_formatted_string_using(
      kLatexPrintMap,*this, short_command_array_);
}

std::string AGraph::GetConsoleString() const {
  return get_formatted_string_using(
      kConsolePrintMap, *this, short_command_array_);
}

std::string AGraph::GetStackString() const {
  std::stringstream print_str; 
  print_str << "---full stack---\n"
            << get_stack_string(*this, command_array_)
            << "---small stack---\n"
            << get_stack_string(*this, short_command_array_); 
  return print_str.str();
}

int AGraph::GetComplexity() const {
  std::vector<bool> commands = GetUtilizedCommands();
  return std::count_if (commands.begin(), commands.end(), [](bool i) {
    return i;
  });
}

void AGraph::ForceRenumberConstants() {
  std::vector<bool> util_commands = GetUtilizedCommands();
  renumber_constants(util_commands, command_array_);
}

int AGraph::Distance(const AGraph &agraph) {
  return (command_array_ != agraph.GetCommandArray()).count();
}

bool AGraph::HasArityTwo(int node) {
  return kIsArity2Map[node];
}

bool AGraph::IsTerminal(int node) {
  return kIsTerminalMap[node];
}

void AGraph::process_modified_command_array() {
  int new_const_number = 0;
  if (!manual_constants_) {
    std::vector<bool> util = GetUtilizedCommands();
    needs_opt_ = check_optimization_requirement(*this, util);
    if (needs_opt_)  {
      new_const_number = renumber_constants(util, command_array_);
    }
  }
  short_command_array_ = backend::SimplifyStack(command_array_);
  num_constants_ = new_const_number;
}

namespace {

bool check_optimization_requirement(
    AGraph &agraph,
    const std::vector<bool> &utilized_commands) {
  Eigen::ArrayX3i command_array = agraph.GetCommandArray();
  for (int i = 0; i < command_array.rows(); i++) {
    if (utilized_commands[i] &&
        command_array(i, ArrayProps::kNodeIdx) == Op::LOAD_C) {
      if (command_array(i, ArrayProps::kOp1) == Op::C_OPTIMIZE ||
          command_array(i, ArrayProps::kOp1) >=
          agraph.GetLocalOptimizationParams().size()) {
        return true;
      }
    }
  }
  return false;
}

int renumber_constants (const std::vector<bool> &utilized_commands,
                        Eigen::ArrayX3i &command_array) {
  int const_num = 0;
  int command_array_depth = command_array.rows();
  for (int i = 0; i < command_array_depth; i++) {
    if (utilized_commands[i] &&
        command_array(i, ArrayProps::kNodeIdx) == Op::LOAD_C) {
      command_array.row(i) << 1, const_num , const_num; 
      const_num ++;
    }
  }
  return const_num;
}

std::string get_formatted_string_using(
    const PrintMap &format_map,
    const AGraph &agraph,
    const Eigen::ArrayX3i &short_command_array){
  std::vector<std::string> string_list;
  for (auto stack_element : short_command_array.rowwise()) {
    std::string temp_string = get_formatted_element_string(
        agraph, stack_element, string_list, format_map);
    string_list.push_back(temp_string);
  }
  return string_list.back();
}

std::string get_formatted_element_string(const AGraph &individual,
                                         const Eigen::ArrayX3i &stack_element,
                                         std::vector<std::string> string_list,
                                         const PrintMap &format_map) {
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

std::string print_string_with_args(const std::string &string,
                                   const std::string &arg1,
                                   const std::string &arg2) {
  std::stringstream stream;
  bool first_found = false;
  for (std::string::const_iterator character = string.begin();
       character != string.end(); character++) {
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

std::string get_stack_string(
    const AGraph &agraph,
    const Eigen::ArrayX3i &command_array) {
  std::string temp_string;
  for (int i = 0; i < command_array.rows(); i++) {
    temp_string += get_stack_element_string(agraph, i, command_array.row(i));
  }
  return temp_string;
}

std::string get_stack_element_string(const AGraph &individual,
                                     int command_index,
                                     const Eigen::ArrayX3i &stack_element) {
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
} // namespace (anonymous)
} // namespace bingo