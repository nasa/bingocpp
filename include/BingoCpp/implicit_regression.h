#ifndef BINGOCPP_INCLUDE_BINGOCPP_IMPLICIT_REGRESSION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_IMPLICIT_REGRESSION_H_

#include <Eigen/Dense>

#include "BingoCpp/equation.h"
#include "BingoCpp/fitness_function.h"
#include "BingoCpp/training_data.h"

namespace bingo {

struct ImplicitTrainingData : TrainingData {
 public:
  Eigen::ArrayXXd x;
  Eigen::ArrayXXd dx_dt;
  ImplicitTrainingData(const Eigen::ArrayXXd &input);
  ImplicitTrainingData(const Eigen::ArrayXXd &input,
                       const Eigen::ArrayXXd &derivative);
  ImplicitTrainingData* GetItem(int item);
  ImplicitTrainingData* GetItem(const std::vector<int> &items);
  int Size() { return x.rows(); }
};

class ImplicitRegression : public VectorBasedFunction {
 public:
  ImplicitRegression(ImplicitTrainingData *training_data, 
                     int required_params = -1,
                     bool normalize_dot = false) :
      VectorBasedFunction(training_data) {
    required_params_ = required_params;
    normalize_dot_ = normalize_dot;
  }

  Eigen::ArrayXXd EvaluateFitnessVector(const Equation &equation);

 private:
  int required_params_;
  bool normalize_dot_;
};

class ImplicitRegressionSchmidt : VectorBasedFunction {

};
} // namespace bingo
#endif //BINGOCPP_INCLUDE_BINGOCPP_IMPLICIT_REGRESSION_H_
