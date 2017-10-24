#ifndef ACYCLIC_GRAPH_HEADER
#define ACYCLIC_GRAPH_HEADER

#include <Eigen/Dense>


typedef std::vector< std::pair<int, std::vector<int> > > CommandStack;

Eigen::ArrayXXd Evaluate(CommandStack stack, Eigen::ArrayXXd x,
                         std::vector<double> constants) ;

Eigen::ArrayXXd SimplifyAndEvaluate(CommandStack stack, Eigen::ArrayXXd x,
                                    std::vector<double> constants) ;
                                    
std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> EvaluateWithDerivative(
  CommandStack stack, Eigen::ArrayXXd x, std::vector<double> constants) ;

void PrintStack(CommandStack stack) ;
CommandStack SimplifyStack(CommandStack stack) ;
std::vector<bool> FindUsedCommands(CommandStack stack) ;

#endif

