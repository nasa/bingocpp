#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <bingocpp/agraph/agraph.h>
#include <bingocpp/agraph/operator_definitions.h>
#include <bingocpp/agraph/string_generation.h>
#include <bingocpp/agraph/evaluation_backend/evaluation_backend.h>
#include <bingocpp/agraph/simplification_backend/simplification_backend.h>
#include <bingocpp/agraph/constants.h>

namespace bingo {

namespace {
const double kFitnessNotSet = 1e9;
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
  return simplification_backend::GetUtilizedCommands(command_array_);
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
  return strm << graph.GetFormattedString("console", false);
}

std::string AGraph::GetFormattedString(std::string format, bool raw){
 if (raw) {
   return string_generation::GetFormattedString(format, this->command_array_, this->constants_);
 }
 return string_generation::GetFormattedString(format, this->command_array_, this->constants_);
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
  short_command_array_ = simplification_backend::SimplifyStack(command_array_);
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

} // namespace bingo