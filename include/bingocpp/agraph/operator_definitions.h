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
#ifndef BINGOCPP_INCLUDE_BINGOCPP_AGRAPH_OPERATOR_DEFINITIONS_H_
#define BINGOCPP_INCLUDE_BINGOCPP_AGRAPH_OPERATOR_DEFINITIONS_H_
#include <string>
#include <vector>
#include <unordered_map>

namespace bingo {

// Operators for CommandArray
enum Op : signed int {
  kInteger=-1,
  kVariable=0,
  kConstant=1,
  kAddition=2,
  kSubtraction=3,
  kMultiplication=4,
  kDivision=5,
  kSin=6,
  kCos=7,
  kExponential=8,
  kLogarithm=9,
  kPower=10,
  kAbs=11,
  kSqrt=12,
  kSafePow=13
};

typedef std::unordered_map<int, std::string> PrintMap;
typedef std::vector<std::vector<std::string>> PrintVector;

const PrintMap kStackPrintMap {
  {2, "({}) + ({})"},
  {3, "({}) - ({})"},
  {4, "({}) * ({})"},
  {5, "({}) / ({}) "},
  {6, "sin ({})"},
  {7, "cos ({})"},
  {8, "exp ({})"},
  {9, "log ({})"},
  {10, "({}) ^ ({})"},
  {11, "abs ({})"},
  {12, "sqrt ({})"},
};

const PrintMap kLatexPrintMap {
  {2, "{} + {}"},
  {3, "{} - ({})"},
  {4, "({})({})"},
  {5, "\\frac{ {} }{ {} }"},
  {6, "sin{ {} }"},
  {7, "cos{ {} }"},
  {8, "exp{ {} }"},
  {9, "log{ {} }"},
  {10, "({})^{ ({}) }"},
  {11, "|{}|"},
  {12, "\\sqrt{ {} }"},
};

const PrintMap kConsolePrintMap {
  {2, "{} + {}"},
  {3, "{} - ({})"},
  {4, "({})({})"},
  {5, "({})/({})"},
  {6, "sin({})"},
  {7, "cos({})"},
  {8, "exp({})"},
  {9, "log({})"},
  {10, "({})^({})"},
  {11, "|{}|"},
  {12, "sqrt({})"},
};

const bool kIsArity2Map[13] = {
  false,
  false,
  true,
  true,
  true,
  true,
  false,
  false,
  false,
  false,
  true,
  false,
  false
};

const bool kIsTerminalMap[13] = {
  true,
  true,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false,
  false
};

const PrintVector kOperatorNames {
  std::vector<std::string> {"load", "x"},
  std::vector<std::string> {"constant", "c"},
  std::vector<std::string> {"add", "addition", "+"},
  std::vector<std::string> {"subtract", "subtraction", "-"},
  std::vector<std::string> {"multiply", "multiplication", "*"},
  std::vector<std::string> {"divide", "division", "/"},
  std::vector<std::string> {"sine", "sin"},
  std::vector<std::string> {"cosine", "cos"},
  std::vector<std::string> {"exponential", "exp", "e"},
  std::vector<std::string> {"logarithm", "log"},
  std::vector<std::string> {"power", "pow", "^"},
  std::vector<std::string> {"absolute value", "||", "|"},
  std::vector<std::string> {"square root", "sqrt"}
};
} // namespace bingo
#endif // BINGOCPP_INCLUDE_BINGOCPP_AGRAPH_OPERATOR_DEFINITIONS_H_
