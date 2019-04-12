#ifndef INCLUDE_BINGOCPP_AGRAPH_H_
#define INCLUDE_BINGOCPP_AGRAPH_H_

#include <set>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Core>

typedef std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvalAndDerivative;
typedef std::unordered_map<int, std::string> PrintMap;

namespace bingo {

class AGraph {
 private:
  Eigen::ArrayX3i command_array_;
  Eigen::VectorXd constants_;
  double fitness_;
  bool fit_set_;
  int genetic_age_;
  // To string operator when passed into stream
  friend std::ostream& operator<<(std::ostream&, const AGraph&);
  static const bool kIsArity2Map[13]; 
  static const bool kIsTerminalMap[13];

 public:
  AGraph();
  AGraph(const AGraph& agraph);
  AGraph copy();
  Eigen::ArrayX3i getCommandArray() const;
  void setCommandArray(Eigen::ArrayX3i command_array);
  void notifyCommandArrayModificiation();
  double getFitness() const;
  void setFitness(double fitness);
  bool isFitnessSet() const;
  void setGeneticAge(const int age);
  int getGeneticAge() const;
  std::vector<bool> getUtilizedCommands() const;
  bool needsLocalOptimization();
  int getNumberLocalOptimizationParams() const;
  void setLocalOptimizationParams(Eigen::VectorXd params);
  Eigen::VectorXd getLocalOptimizationParams() const;
  Eigen::ArrayXXd evaluateEquationAt(Eigen::ArrayXXd& x);
  EvalAndDerivative evaluateEquationWithXGradientAt(Eigen::ArrayXXd& x);
  EvalAndDerivative evaluateEquationWithLocalOptGradientAt(Eigen::ArrayXXd& x);
  std::string getLatexString() const;
  std::string getConsoleString() const;
  std::string getStackString() const;
  int getComplexity() const;
};
} // namespace bingo
#endif //INCLUDE_BINGOCPP_AGRAPH_H_
