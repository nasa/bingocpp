/*
 * Copyright 2018 United States Government as represented by the Administrator
 * of the National Aeronautics and Space Administration. No copyright is claimed
 * in the United States under Title 17, U.S. Code. All Other Rights Reserved.
 *
 * The Bingo Mini-app platform is licensed under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with the
 * License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
*/
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

  Eigen::ArrayXXd EvaluateFitnessVector(Equation &equation) const;

 private:
  int required_params_;
  bool normalize_dot_;
  static const int kNoneRequired = -1;
};

class ImplicitRegressionSchmidt : VectorBasedFunction {

};
} // namespace bingo
#endif //BINGOCPP_INCLUDE_BINGOCPP_IMPLICIT_REGRESSION_H_
