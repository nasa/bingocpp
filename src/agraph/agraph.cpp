#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <bingocpp/agraph/agraph.h>
#include <bingocpp/agraph/operator_definitions.h>
#include <bingocpp/agraph/evaluation_backend/evaluation_backend.h>
#include <bingocpp/agraph/constants.h>

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

std::string get_stack_string(const Eigen::ArrayX3i &command_array,
                             const Eigen::VectorXd &constants);

std::string get_stack_element_string(const Eigen::VectorXd &constants,
                                     int command_index,
                                     const Eigen::ArrayX3i &stack_element);
} // namespace

AGraph::AGraph() {
  command_array_ = Eigen::ArrayX3i(0, 3);
  short_command_array_ = Eigen::ArrayX3i(0, 3);
  constants_ = Eigen::VectorXd(0);
  needs_opt_ = false;
  num_constants_ = 0;
  fitness_ = kFitnessNotSet;
  fit_set_ = false;
  genetic_age_ = 0;
  modified_ = false;
}
  
AGraph::AGraph(const AGraph &agraph) {
  command_array_ = agraph.command_array_;
  short_command_array_ = agraph.short_command_array_;
  constants_ = agraph.constants_;
  needs_opt_ = agraph.needs_opt_;
  num_constants_ = agraph.num_constants_;
  fitness_ = agraph.fitness_;
  fit_set_ = agraph.fit_set_;
  genetic_age_ = agraph.genetic_age_;
  modified_ = agraph.modified_;
}

AGraph::AGraph(const AGraphState &state) {
  command_array_ = std::get<0>(state);
  short_command_array_ = std::get<1>(state);
  constants_ = std::get<2>(state);
  needs_opt_ = std::get<3>(state);
  num_constants_ = std::get<4>(state);
  fitness_ = std::get<5>(state);
  fit_set_ = std::get<6>(state);
  genetic_age_ = std::get<7>(state);
  modified_ = std::get<8>(state);
}

AGraph AGraph::Copy() {
  return AGraph(*this);
}

AGraphState AGraph::DumpState() {
  return AGraphState(command_array_, short_command_array_, constants_,
                     needs_opt_, num_constants_, fitness_, fit_set_,
                     genetic_age_, modified_);;
}

const Eigen::ArrayX3i &AGraph::GetCommandArray() const {
  return command_array_;
}

Eigen::ArrayX3i &AGraph::GetCommandArrayModifiable() {
  notify_agraph_modification();
  return command_array_;
}

void AGraph::SetCommandArray(const Eigen::ArrayX3i &command_array) {
  command_array_ = command_array;
  notify_agraph_modification();
}

void AGraph::notify_agraph_modification() {
  fitness_ = kFitnessNotSet;
  fit_set_ = false;
  modified_ = true;
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
  return evaluation_backend::GetUtilizedCommands(command_array_);
}

bool AGraph::NeedsLocalOptimization() {
  if (modified_) {
      process_modified_command_array();
  }
  return needs_opt_;
}

int AGraph::GetNumberLocalOptimizationParams() {
  if (modified_) {
      process_modified_command_array();
  }
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
AGraph::EvaluateEquationAt(const Eigen::ArrayXXd &x) {
  if (modified_) {
      process_modified_command_array();
  }
  Eigen::ArrayXXd f_of_x; 
  try {
    f_of_x = evaluation_backend::Evaluate(this->short_command_array_,
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
AGraph::EvaluateEquationWithXGradientAt(const Eigen::ArrayXXd &x) {
  if (modified_) {
      process_modified_command_array();
  }
  EvalAndDerivative df_dx;
  try {
    df_dx = evaluation_backend::EvaluateWithDerivative(this->short_command_array_,
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
AGraph::EvaluateEquationWithLocalOptGradientAt(const Eigen::ArrayXXd &x) {
  if (modified_) {
      process_modified_command_array();
  }
  EvalAndDerivative df_dc;
  try {
    df_dc = evaluation_backend::EvaluateWithDerivative(this->short_command_array_,
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

std::ostream &operator<<(std::ostream &strm, AGraph &graph) {
  return strm << graph.GetConsoleString();
}

std::string AGraph::GetLatexString() {
  if (modified_) {
      process_modified_command_array();
  }
  return get_formatted_string_using(
      kLatexPrintMap,*this, short_command_array_);
}

std::string AGraph::GetConsoleString() {
  if (modified_) {
      process_modified_command_array();
  }
  return get_formatted_string_using(
      kConsolePrintMap, *this, short_command_array_);
}

std::string AGraph::GetStackString() {
  if (modified_) {
      process_modified_command_array();
  }
  std::stringstream print_str;
  Eigen::VectorXd empty_consts;
  print_str << "---full stack---\n"
            << get_stack_string(command_array_, empty_consts)
            << "---small stack---\n"
            << get_stack_string(short_command_array_, constants_);
  return print_str.str();
}

int AGraph::GetComplexity() const {
  std::vector<bool> commands = GetUtilizedCommands();
  return std::count_if (commands.begin(), commands.end(), [](bool i) {
    return i;
  });
}

int AGraph::Distance(const AGraph &agraph) {
  return (command_array_ != agraph.GetCommandArray()).count();
}

bool AGraph::HasArityTwo(int node) {
  return kIsArity2Map.at(node);
}

bool AGraph::IsTerminal(int node) {
  return kIsTerminalMap.at(node);
}

void AGraph::process_modified_command_array() {
  short_command_array_ = evaluation_backend::SimplifyStack(command_array_);
  int new_const_number = 0;
  for (int i = 0; i < short_command_array_.rows(); i++) {
    if (short_command_array_(i, kOpIdx) == Op::kConstant) {
      short_command_array_.row(i) << Op::kConstant, new_const_number, new_const_number;
      new_const_number ++;
    }
  }

  int optimization_aggression = 0;
  if (optimization_aggression == 0 && new_const_number <= num_constants_) {
    constants_.conservativeResize(new_const_number);
  } else if (optimization_aggression == 1 && new_const_number == num_constants_) {
    // reuse old constants
  } else {
    constants_.resize(new_const_number);
    constants_.setOnes(new_const_number);
    if (new_const_number > 0) {
      needs_opt_ = true;
    }
  }
  modified_ = false;
  num_constants_ = new_const_number;
}

namespace {

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
  int node = stack_element(0, kOpIdx);
  int param1 = stack_element(0, kParam1Idx);
  int param2 = stack_element(0, kParam2Idx);

  std::string temp_string;
  if (node == Op::kVariable) {
    temp_string = "X_" + std::to_string(param1);
  } else if (node == Op::kConstant) {
    if (param1 == kOptimizeConstant ||
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
    const Eigen::ArrayX3i &command_array,
    const Eigen::VectorXd &constants) {
  std::string temp_string;
  for (int i = 0; i < command_array.rows(); i++) {
    temp_string += get_stack_element_string(constants, i, command_array.row(i));
  }
  return temp_string;
}

std::string get_stack_element_string(const Eigen::VectorXd &constants,
                                     int command_index,
                                     const Eigen::ArrayX3i &stack_element) {
  int node = stack_element(0, kOpIdx);
  int param1 = stack_element(0, kParam1Idx);
  int param2 = stack_element(0, kParam2Idx);

  std::string temp_string = "("+ std::to_string(command_index) +") <= ";
  if (node == Op::kVariable) {
    temp_string += "X_" + std::to_string(param1);
  } else if (node == Op::kConstant) {
    if (param1 == kOptimizeConstant ||
        param1 >= constants.size()) {
      temp_string += "C";
    } else {
      temp_string += "C_" + std::to_string(param1) + " = " + 
                     std::to_string(constants[param1]);
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