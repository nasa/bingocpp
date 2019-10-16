#ifndef BINGOCPP_INCLUDE_BINGOCPP_EXPLICIT_REGRESSION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_EXPLICIT_REGRESSION_H_

#include <string>
#include <vector>

#include <Eigen/Core>

#include "BingoCpp/equation.h"
#include "BingoCpp/fitness_function.h"
#include "BingoCpp/training_data.h"

namespace bingo {

struct ExplicitTrainingData : TrainingData {
  Eigen::ArrayXXd x;

  Eigen::ArrayXXd y;

  ExplicitTrainingData(const Eigen::ArrayXXd &input,
                       const Eigen::ArrayXXd &output) {
    x = input;
    y = output;
  }

  ExplicitTrainingData(const ExplicitTrainingData &other) {
    x = other.x;
    y = other.y;
  }

  ~ExplicitTrainingData() { }

  ExplicitTrainingData *GetItem(int item);

  ExplicitTrainingData *GetItem(const std::vector<int> &items);

  int Size() {
    return x.rows();
  }
};

class ExplicitRegression : public VectorBasedFunction {
 public:
  ExplicitRegression(ExplicitTrainingData *training_data,
                     std::string metric="mae") : 
      VectorBasedFunction(new ExplicitTrainingData(*training_data), metric) {}

  ~ExplicitRegression() {
    delete training_data_;
  }

  Eigen::ArrayXXd EvaluateFitnessVector(const Equation &individual) const;
};
} // namespace bingo
#endif // BINGOCPP_INCLUDE_BINGOCPP_EXPLICIT_REGRESSION_H_
