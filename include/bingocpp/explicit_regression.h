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
#ifndef BINGOCPP_INCLUDE_BINGOCPP_EXPLICIT_REGRESSION_H_
#define BINGOCPP_INCLUDE_BINGOCPP_EXPLICIT_REGRESSION_H_

#include <string>
#include <vector>

#include <Eigen/Core>

#include "bingocpp/equation.h"
#include "bingocpp/fitness_function.h"
#include "bingocpp/training_data.h"

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

  Eigen::ArrayXXd EvaluateFitnessVector(Equation &individual) const;
};
} // namespace bingo
#endif // BINGOCPP_INCLUDE_BINGOCPP_EXPLICIT_REGRESSION_H_
