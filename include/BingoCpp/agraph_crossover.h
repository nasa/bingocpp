#ifndef INCLUDE_BINGOCPP_AGRAPH_CROSSOVER_H_
#define INCLUDE_BINGOCPP_AGRAPH_CROSSOVER_H_

#include <BingoCpp/agraph.h>
#include <BingoCpp/constants.h>

namespace bingo {

typedef std::pair<AGraph, AGraph> CrossoverChildren;

class AGraphCrossover {
 public:
  AGraphCrossover();
  AGraphCrossover(const AGraphCrossover& crossover) = delete;
  CrossoverChildren crossover(AGraph& parent_1, AGraph& parent_2);
};
} // namespace bingo
#endif //INCLUDE_BINGOCPP_AGRAPH_CROSSOVER_H_