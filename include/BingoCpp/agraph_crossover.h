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
#ifndef INCLUDE_BINGOCPP_AGRAPH_CROSSOVER_H_
#define INCLUDE_BINGOCPP_AGRAPH_CROSSOVER_H_

#include <random>

#include <BingoCpp/agraph.h>
#include <BingoCpp/constants.h>

namespace bingo {

typedef std::pair<AGraph, AGraph> CrossoverChildren;

/**
 * @brief The Class that perfroms Crossover among acyclic graphs.
 * 
 * This ontains the implementation of single point crossover between
 * acyclic graph individuals.
 */
class AGraphCrossover {
 public:
  AGraphCrossover();

  AGraphCrossover(std::mt19937::result_type seed);
  
  AGraphCrossover(const AGraphCrossover& crossover);
  /**
   * @brief Crossover between acyclic graph individuals
   * 
   * @param parent_1 The first parent individual
   * @param parent_2 The second parent individual
   * @return CrossoverChildren The two children from the crossover
   */
  CrossoverChildren crossover(AGraph& parent_1, AGraph& parent_2);

 private:
  std::mt19937 engine_;
};
} // namespace bingo
#endif //INCLUDE_BINGOCPP_AGRAPH_CROSSOVER_H_