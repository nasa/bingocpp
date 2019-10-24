#ifndef BINGOCPP_INCLUDE_BINGOCPP_IMPLICIT_REGRESSION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_IMPLICIT_REGRESSION_H_

#include <Eigen/Dense>

#include "BingoCpp/equation.h"
#include "BingoCpp/fitness_function.h"
#include "BingoCpp/training_data.h"

#include "BingoCpp/utils.h"

namespace bingo {

struct ImplicitTrainingData : TrainingData {
 public:
  Eigen::ArrayXXd x;

  Eigen::ArrayXXd dx_dt;

  ImplicitTrainingData(const Eigen::ArrayXXd &input) {
    InputAndDeriviative input_and_deriv = CalculatePartials(input);
    x = input_and_deriv.first;
    dx_dt = input_and_deriv.second;
  }

  ImplicitTrainingData(const Eigen::ArrayXXd &input,
                       const Eigen::ArrayXXd &derivative) {
    x = input; 
    dx_dt = derivative;
  }

  ImplicitTrainingData(ImplicitTrainingData &other) {
    x = other.x;
    dx_dt = other.dx_dt;
  } 

  ImplicitTrainingData* GetItem(int item);

  ImplicitTrainingData* GetItem(const std::vector<int> &items);

  int Size() { 
    return x.rows();
  }
};

class ImplicitRegression : public VectorBasedFunction {
 public:
  ImplicitRegression(ImplicitTrainingData *training_data, 
                     int required_params = kNoneRequired,
                     bool normalize_dot = false,
                     std::string metric="mae") :
      VectorBasedFunction(new ImplicitTrainingData(*training_data), metric) {
    required_params_ = required_params;
    normalize_dot_ = normalize_dot;
  }

  ~ImplicitRegression() {
    delete training_data_;
  }

  Eigen::ArrayXXd EvaluateFitnessVector(const Equation &equation) const;

 private:
  int required_params_;
  bool normalize_dot_;
  static const int kNoneRequired = -1;
};

class ImplicitRegressionSchmidt : VectorBasedFunction {

};
} // namespace bingo
#endif //BINGOCPP_INCLUDE_BINGOCPP_IMPLICIT_REGRESSION_H_
