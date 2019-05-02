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
#ifndef INCLUDE_BINGOCPP_CONSTANTS_H_
#define INCLUDE_BINGOCPP_CONSTANTS_H_

namespace bingo {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// Properties for CommandArray
enum ArrayProps : unsigned int {
  kNodeIdx = 0,
  kOp1 = 1,
  kOp2 = 2,
  kArrayCols = 3
};

// Operators for CommandArray
enum Op : signed int {
  LOAD_X = 0,
  LOAD_C = 1,
  C_OPTIMIZE = -1,
  ADD = 2,
  SUB = 3,
  MULT = 4,
  DIV = 5,
  SIN = 6,
  COS = 7,
  EXP = 8,
  LOG = 9,
  POW = 10,
  ABS = 11,
  SQRT = 12
};
}
#endif
